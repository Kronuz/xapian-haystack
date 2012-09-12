# Copyright (C) 2012 German M. Bravo
# Copyright (C) 2009, 2010, 2011, 2012 David Sauve
# Copyright (C) 2009, 2010 Trapeze

__author__ = 'David Sauve'
__version__ = (2, 0, 0)

import time
import datetime
import cPickle as pickle
import os
import re
import shutil
import sys
from threading import local
from math import sin, cos, asin, degrees, radians, pi

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.encoding import force_unicode
from haystack.utils.geo import Point

from haystack import connections
from haystack.backends import BaseEngine, BaseSearchBackend, BaseSearchQuery, SearchNode, log_query
from haystack.constants import ID
from haystack.exceptions import HaystackError, MissingDependency
from haystack.models import SearchResult
from haystack.utils import get_identifier, normalize

try:
    import xapian
except ImportError:
    raise MissingDependency("The 'xapian' backend requires the installation of 'xapian'. Please refer to the documentation.")


DOCUMENT_ID_TERM_PREFIX = 'Q'
DOCUMENT_CUSTOM_TERM_PREFIX = 'X'
DOCUMENT_CT_TERM_PREFIX = DOCUMENT_CUSTOM_TERM_PREFIX + 'CONTENTTYPE'

MEMORY_DB_NAME = ':memory:'

DEFAULT_XAPIAN_FLAGS = (
    xapian.QueryParser.FLAG_PHRASE |
    xapian.QueryParser.FLAG_BOOLEAN |
    xapian.QueryParser.FLAG_LOVEHATE |
    xapian.QueryParser.FLAG_WILDCARD |
    xapian.QueryParser.FLAG_PURE_NOT
)


g_tl = local()


EARTH_RADIUS = 6371000.0  # in meters
MIN_LAT = radians(-90)
MAX_LAT = radians(90)
MIN_LON = radians(0)
MAX_LON = radians(360)


def bounding_box(lat, lon, distance):
    lat = radians(lat)
    lon = radians(lon)

    if lat < MIN_LAT or lat > MAX_LAT:
        raise

    while lon < MIN_LON:
        lon += 2 * pi
    while lon > MAX_LON:
        lon -= 2 * pi

    dlat = distance / EARTH_RADIUS

    min_lat = lat - dlat
    max_lat = lat + dlat

    min_lat, max_lat = min(min_lat, max_lat), max(min_lat, max_lat)

    if min_lat > MIN_LAT and max_lat < MAX_LAT:
        dlon = asin(sin(dlat) / cos(lat))
        min_lon = lon - dlon
        max_lon = lon + dlon
        while min_lon < MIN_LON:
            min_lon += 2 * pi
        while max_lon > MAX_LON:
            max_lon -= 2 * pi
    else:
        # a pole is within the distance
        min_lat = max(min_lat, MIN_LAT)
        max_lat = min(max_lat, MAX_LAT)
        min_lon = MIN_LON
        max_lon = MAX_LON
    return degrees(min_lat), degrees(min_lon), degrees(max_lat), degrees(max_lon)


def _database(self, writable=False):
    """
    Private method that returns a xapian.Database for use.
    Optional arguments:
        ``writable`` -- Open the database in read/write mode (default=False)
    Returns an instance of a xapian.Database or xapian.WritableDatabase
    """
    database_key = '%s:%s:%s:%s' % (getattr(self, 'path', ''), getattr(self, 'host', ''), getattr(self, 'port', ''), writable)

    if not hasattr(g_tl, 'databases'):
        g_tl.databases = {}

    database = g_tl.databases.get(database_key)
    if not database:
        if hasattr(self, 'path'):
            if writable:
                database = xapian.WritableDatabase(self.path, xapian.DB_CREATE_OR_OPEN)
            else:
                try:
                    database = xapian.Database(self.path)
                except xapian.DatabaseOpeningError:
                    try:
                        database = xapian.WritableDatabase(self.path, xapian.DB_CREATE_OR_OPEN)
                        database.close()
                        database = xapian.Database(self.path)
                    except xapian.DatabaseOpeningError:
                        raise InvalidIndexError(u'Unable to open index at %s' % self.path)
        else:
            if writable:
                database = xapian.remote_open_writable(self.host, self.port, self.timeout)
            else:
                database = xapian.remote_open(self.host, self.port, self.timeout)
        g_tl.databases[database_key] = database
    else:
        if not writable:  # Make sure we always read the latest:
            # print '_database.reopen()'
            database.reopen()
    return database


def index_document(self, documents, commit=True):
    database = _database(self, writable=True)
    for document_id, document_values, document_terms, document_text, document_data, language in documents:
        document = xapian.Document()
        term_generator = xapian.TermGenerator()
        term_generator.set_database(database)
        term_generator.set_stemmer(xapian.Stem(language))
        if self.include_spelling is True:
            term_generator.set_flags(xapian.TermGenerator.FLAG_SPELLING)
        term_generator.set_document(document)
        for values in document_values:
            document.add_value(*values)
        for terms in document_terms:
            document.add_term(*terms)
        for text in document_text:
            term_generator.index_text(*text)
        for data in document_data:
            document.set_data(*data)
        database.replace_document(document_id, document)
    if commit:
        commit_database(self)
    return True


def remove_document(self, documents, commit=True):
    database = _database(self, writable=True)
    for document_id in documents:
        database.delete_document(document_id)
    if commit:
        commit_database(self)
    return True


def commit_database(self):
    database = _database(self, writable=True)
    database.commit()
    return True


class InvalidIndexError(HaystackError):
    """Raised when an index can not be opened."""
    pass


class XHValueRangeProcessor(xapian.ValueRangeProcessor):
    def __init__(self, backend):
        # FIXME: This needs to get smarter about pulling the right backend.
        self.backend = backend or XapianSearchBackend()
        xapian.ValueRangeProcessor.__init__(self)

    def __call__(self, begin, end):
        """
        Construct a tuple for value range processing.
        `begin` -- a string in the format '<field_name>:[low_range]'
        If 'low_range' is omitted, assume the smallest possible value.
        `end` -- a string in the the format '[high_range|*]'. If '*', assume
        the highest possible value.
        Return a tuple of three strings: (column, low, high)
        """
        colon = begin.find(':')
        field_name = begin[:colon]
        begin = begin[colon + 1:len(begin)]
        for field_dict in self.backend.schema:
            if field_dict['field_name'] == field_name:
                if not begin:
                    if field_dict['type'] == 'text':
                        begin = u'a'  # TODO: A better way of getting a min text value?
                    elif field_dict['type'] == 'long':
                        begin = -sys.maxint - 1
                    elif field_dict['type'] == 'float':
                        begin = float('-inf')
                    elif field_dict['type'] == 'date' or field_dict['type'] == 'datetime':
                        begin = u'00010101000000'
                elif end == '*':
                    if field_dict['type'] == 'text':
                        end = u'z' * 100  # TODO: A better way of getting a max text value?
                    elif field_dict['type'] == 'long':
                        end = sys.maxint
                    elif field_dict['type'] == 'float':
                        end = float('inf')
                    elif field_dict['type'] == 'date' or field_dict['type'] == 'datetime':
                        end = u'99990101000000'
                if field_dict['type'] == 'float':
                    begin = _marshal_value(float(begin))
                    end = _marshal_value(float(end))
                elif field_dict['type'] == 'long':
                    begin = _marshal_value(long(begin))
                    end = _marshal_value(long(end))
                return field_dict['column'], str(begin), str(end)


class XHExpandDecider(xapian.ExpandDecider):
    def __call__(self, term):
        """
        Return True if the term should be used for expanding the search
        query, False otherwise.

        Currently, we only want to ignore terms beginning with `DOCUMENT_CT_TERM_PREFIX`
        """
        if term.startswith(DOCUMENT_CT_TERM_PREFIX):
            return False
        return True


class XapianSearchBackend(BaseSearchBackend):
    """
    `SearchBackend` defines the Xapian search backend for use with the Haystack
    API for Django search.

    It uses the Xapian Python bindings to interface with Xapian, and as
    such is subject to this bug: <http://trac.xapian.org/ticket/364> when
    Django is running with mod_python or mod_wsgi under Apache.

    Until this issue has been fixed by Xapian, it is neccessary to set
    `WSGIApplicationGroup to %{GLOBAL}` when using mod_wsgi, or
    `PythonInterpreter main_interpreter` when using mod_python.

    In order to use this backend, either `HAYSTACK_XAPIAN_PATH` or
    `HAYSTACK_XAPIAN_HOST` and `HAYSTACK_XAPIAN_PORT` must be set in
    your settings.  This should point to a location where you would your
    indexes to reside.
    """

    inmemory_db = None

    def __init__(self, connection_alias, language=None, **connection_options):
        """
        Instantiates an instance of `SearchBackend`.

        Optional arguments:
            `connection_alias` -- The name of the connection
            `**connection_options` -- The various options needed to setup
              the backend.
        """
        super(XapianSearchBackend, self).__init__(connection_alias, **connection_options)

        if 'PATH' in connection_options:
            self.path = connection_options['PATH']
            if self.path != MEMORY_DB_NAME and \
               not os.path.exists(self.path):
                os.makedirs(self.path)
                if not os.access(self.path, os.W_OK):
                    raise IOError("The path to your Xapian index '%s' is not writable for the current user/group." % self.path)
        elif 'HOST' not in connection_options or not 'PORT' not in connection_options:
            raise ImproperlyConfigured("You must specify 'PATH' or 'HOST' and 'PORT' in your settings for connection '%s'." % connection_alias)
        else:
            self.host = connection_options['HOST']
            self.port = connection_options['PORT']
            self.timeout = connection_options.get('TIMEOUT', 0)

        self.weighting_scheme = connection_options.get('WEIGHTING_SCHEME')
        self.flags = connection_options.get('FLAGS', DEFAULT_XAPIAN_FLAGS)
        self.language = language or getattr(settings, 'HAYSTACK_XAPIAN_LANGUAGE', 'english')
        self._schema = None
        self._content_field_name = None

    @property
    def schema(self):
        if not self._schema:
            self._content_field_name, self._schema = self.build_schema(connections[self.connection_alias].get_unified_index().all_searchfields())
        return self._schema

    @property
    def content_field_name(self):
        if not self._content_field_name:
            self._content_field_name, self._schema = self.build_schema(connections[self.connection_alias].get_unified_index().all_searchfields())
        return self._content_field_name

    def update(self, index, iterable, commit=True):
        """
        Updates the `index` with any objects in `iterable` by adding/updating
        the database as needed.

        Required arguments:
            `index` -- The `SearchIndex` to process
            `iterable` -- An iterable of model instances to index

        For each object in `iterable`, a document is created containing all
        of the terms extracted from `index.full_prepare(obj)` with field prefixes,
        and 'as-is' as needed.  Also, if the field type is 'text' it will be
        stemmed and stored with the 'Z' prefix as well.

        eg. `content:Testing` ==> `testing, Ztest, ZXCONTENTtest, XCONTENTtest`

        Each document also contains an extra term in the format:

        `XCONTENTTYPE<app_name>.<model_name>`

        As well as a unique identifier in the the format:

        `Q<app_name>.<model_name>.<pk>`

        eg.: foo.bar (pk=1) ==> `Qfoo.bar.1`, `XCONTENTTYPEfoo.bar`

        This is useful for querying for a specific document corresponding to
        a model instance.

        The document also contains a pickled version of the object itself and
        the document ID in the document data field.

        Finally, we also store field values to be used for sorting data.  We
        store these in the document value slots (position zero is reserver
        for the document ID).  All values are stored as unicode strings with
        conversion of float, int, double, values being done by Xapian itself
        through the use of the :method:xapian.sortable_serialise method.
        """
        try:
            documents = []
            for obj in iterable:
                document_id = DOCUMENT_ID_TERM_PREFIX + get_identifier(obj)
                data = index.full_prepare(obj)
                document_values = []
                document_terms = []
                document_text = []
                document_data = []
                weights = index.get_field_weights()
                for field in self.schema:
                    if field['field_name'] in data.keys():
                        prefix = DOCUMENT_CUSTOM_TERM_PREFIX + field['field_name'].upper()
                        value = data[field['field_name']]
                        if not field['stored']:
                            del data[field['field_name']]
                        if field['type'] == 'geo_point':
                            if value:
                                lat, lng = map(float, value.split(','))
                                value = xapian.LatLongCoord(lat, lng).serialise()
                                data[field['field_name']] = Point(lng, lat)
                                document_values.append([field['column'], value])
                                for term, weight in [(value[:-i], 5 - i) if i else (value, 5) for i in range(5)]:
                                    document_terms.append([DOCUMENT_CUSTOM_TERM_PREFIX + field['field_name'].upper() + term, weight])
                        elif field['multi_valued'] == 'false':
                            document_values.append([field['column'], _marshal_value(value)])
                            value = [value]
                        try:
                            weight = int(weights[field['field_name']])
                        except KeyError:
                            weight = 1
                        if field['type'] == 'text':
                            for term in value:
                                if term:
                                    if field['mode'] == 'autocomplete':  # mode = content, autocomplete, tagged
                                        terms = _marshal_weighted_terms({term: weight})
                                        for term, weight in terms.items():
                                            document_terms.append([DOCUMENT_CUSTOM_TERM_PREFIX + 'AC' + term, weight])
                                    elif field['mode'] == 'tagged':
                                        terms = _marshal_tagged_terms({term: weight})
                                        for term, weight in terms.items():
                                            document_terms.append([DOCUMENT_CUSTOM_TERM_PREFIX + 'TAG' + term, weight])
                                    else:
                                        term = _marshal_term(term)
                                        if self.content_field_name == field['field_name']:
                                            document_text.append([term, weight])
                                        document_text.append([term, weight, prefix])
                        elif field['type'] != 'geo_point':
                            for term in value:
                                term = _marshal_term(term)
                                if self.content_field_name == field['field_name']:
                                    document_terms.append([term, weight])
                                document_terms.append([prefix + term, weight])

                document_data.append([pickle.dumps(
                    (obj._meta.app_label, obj._meta.module_name, obj.pk, data),
                    pickle.HIGHEST_PROTOCOL
                )])
                document_terms.append([document_id])
                document_terms.append([
                    DOCUMENT_CT_TERM_PREFIX + u'%s.%s' %
                    (obj._meta.app_label, obj._meta.module_name)
                ])
                document = (document_id, document_values, document_terms, document_text, document_data, self.language,)
                index_document(self, [document], commit=False)
            if commit:
                commit_database(self)

        except UnicodeDecodeError:
            sys.stderr.write('Chunk failed.\n')
            pass

    def remove(self, obj, commit=True):
        """
        Remove indexes for `obj` from the database.

        We delete all instances of `Q<app_name>.<model_name>.<pk>` which
        should be unique to this object.
        """
        documents = []
        documents.append(DOCUMENT_ID_TERM_PREFIX + get_identifier(obj))
        remove_document(self, documents, commit=commit)

    def clear(self, models=[]):
        """
        Clear all instances of `models` from the database or all models, if
        not specified.

        Optional Arguments:
            `models` -- Models to clear from the database (default = [])

        If `models` is empty, an empty query is executed which matches all
        documents in the database.  Afterwards, each match is deleted.

        Otherwise, for each model, a `delete_document` call is issued with
        the term `XCONTENTTYPE<app_name>.<model_name>`.  This will delete
        all documents with the specified model type.
        """
        if not models and self.path:
            # Because there does not appear to be a "clear all" method,
            # it's much quicker to remove the contents of the `self.path`
            # folder than it is to remove each document one at a time.
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
        else:
            documents = []
            for model in models:
                documents.append(DOCUMENT_CT_TERM_PREFIX + '%s.%s' % (model._meta.app_label, model._meta.module_name))
            remove_document(self, documents)

    def document_count(self):
        try:
            return self._database().get_doccount()
        except InvalidIndexError:
            return 0

    @log_query
    def search(self, query, **kwargs):
        """
        Executes the Xapian::query as defined in `query`.

        Required arguments:
            `query` -- Search query to execute

        Optional arguments:
            `sort_by` -- Sort results by specified field (default = None)
            `start_offset` -- Slice results from `start_offset` (default = 0)
            `end_offset` -- Slice results at `end_offset` (default = None), if None, then all documents
            `fields` -- Filter results on `fields` (default = '')
            `highlight` -- Highlight terms in results (default = False)
            `facets` -- Facet results on fields (default = None)
            `date_facets` -- Facet results on date ranges (default = None)
            `query_facets` -- Facet results on queries (default = None)
            `narrow_queries` -- Narrow queries (default = None)
            `spelling_query` -- An optional query to execute spelling suggestion on
            `limit_to_registered_models` -- Limit returned results to models registered in the current `SearchSite` (default = True)
            `within` --  Rectangular area inside two points within which to limit the search (default = None)
            `dwithin` --  Circular area of a distance around a center point within which to limit the search (default = None)

        Returns:
            A dictionary with the following keys:
                `results` -- A list of `SearchResult`
                `hits` -- The total available results
                `facets` - A dictionary of facets with the following keys:
                    `fields` -- A list of field facets
                    `dates` -- A list of date facets
                    `queries` -- A list of query facets
            If faceting was not used, the `facets` key will not be present

        If `query` is None, returns no results.

        If `INCLUDE_SPELLING` was enabled in the connection options, the
        extra flag `FLAG_SPELLING_CORRECTION` will be passed to the query parser
        and any suggestions for spell correction will be returned as well as
        the results.
        """
        if query.empty():
            return {
                'results': [],
                'hits': 0,
            }

        additional_fields = {}

        database = self._database()

        result_class = kwargs.get('result_class')
        models = kwargs.get('models')
        sort_by = kwargs.get('sort_by')
        start_offset = kwargs.get('start_offset', 0)
        end_offset = kwargs.get('end_offset')
        highlight = kwargs.get('highlight', False)
        facets = kwargs.get('facets')
        date_facets = kwargs.get('date_facets')
        query_facets = kwargs.get('query_facets')
        spelling_query = kwargs.get('spelling_query')
        narrow_queries = kwargs.get('narrow_queries')
        limit_to_registered_models = kwargs.get('limit_to_registered_models', True)
        within = kwargs.get('within')
        dwithin = kwargs.get('dwithin')
        distance_point = kwargs.get('distance_point')

        if kwargs.get('result_class'):
            result_class = kwargs['result_class']

        if result_class is None:
            result_class = SearchResult

        if self.include_spelling is True:
            spelling_suggestion = self._do_spelling_suggestion(database, query, spelling_query)
        else:
            spelling_suggestion = ''

        if narrow_queries is not None:
            query = xapian.Query(
                xapian.Query.OP_AND, query,
                xapian.Query(
                    xapian.Query.OP_AND, [self.parse_query(narrow_query) for narrow_query in narrow_queries]
                )
            )

        if models and len(models):
            registered_models = sorted(['%s.%s' % (model._meta.app_label, model._meta.module_name) for model in models])
        elif limit_to_registered_models:
            registered_models = self.build_models_list()
        else:
            registered_models = []

        if len(registered_models) > 0:
            query = xapian.Query(
                xapian.Query.OP_AND, query,
                xapian.Query(
                    xapian.Query.OP_OR, [
                        xapian.Query('%s%s' % (DOCUMENT_CT_TERM_PREFIX, model)) for model in registered_models
                    ]
                )
            )

        def get_terms_lat_long_distance(center, distance):
            terms = set([center.serialise()[:-4]])
            min_lat, min_lon, max_lat, max_lon = bounding_box(center.latitude, center.longitude, distance)
            for lat in range(int(min_lat), 1 + int(max_lat)):
                for lon in range(int(min_lon), 1 + int(max_lon)):
                    metric = xapian.GreatCircleMetric()
                    coord = xapian.LatLongCoord(lat, lon)
                    term = coord.serialise()[:-4]
                    if term not in terms:
                        if xapian.LatLongMetric.pointwise_distance(metric, center, coord) < distance:
                            terms.add(term)
                        elif xapian.LatLongMetric.pointwise_distance(metric, center, xapian.LatLongCoord(lat + 1, lon)) < distance:
                            terms.add(term)
                        elif xapian.LatLongMetric.pointwise_distance(metric, center, xapian.LatLongCoord(lat, lon + 1)) < distance:
                            terms.add(term)
                        elif xapian.LatLongMetric.pointwise_distance(metric, center, xapian.LatLongCoord(lat + 1, lon + 1)) < distance:
                            terms.add(term)
                    if len(terms) > 1000:
                        return
            return terms

        if within is not None:
            from haystack.utils.geo import generate_bounding_box

            ((min_lat, min_lng), (max_lat, max_lng)) = generate_bounding_box(within['point_1'], within['point_2'])

            #LatLongBoundingPostingSource
            raise NotImplemented

        if dwithin is not None:
            lng, lat = dwithin['point'].get_coords()
            distance = dwithin['distance'].m

            k1, k2 = 1000.0, 1.0  # `k1` and `k2` control how the weights varies with distance.
            metric = xapian.GreatCircleMetric()
            coords = xapian.LatLongCoords()
            center = xapian.LatLongCoord(lat, lng)
            coords.append(center)

            query_list = []
            terms = get_terms_lat_long_distance(center, distance)
            if terms is not None:
                for term in terms:
                    query_list.append(xapian.Query('%s%s%s' % (DOCUMENT_CUSTOM_TERM_PREFIX, dwithin['field'].upper(), term)))
                query = xapian.Query(
                    xapian.Query.OP_AND, query,
                    xapian.Query(xapian.Query.OP_OR, query_list)
                )

            query = xapian.Query(
                xapian.Query.OP_FILTER, query,
                xapian.Query(
                    xapian.LatLongDistancePostingSource(self._value_column(dwithin['field']), coords, metric, distance, k1, k2)
                )
            )

        enquire = xapian.Enquire(database)
        if self.weighting_scheme:
            enquire.set_weighting_scheme(xapian.BM25Weight(*self.weighting_scheme))
        enquire.set_query(query)

        if sort_by:
            _sort_by, sort_by = sort_by, []
            for sort_field in _sort_by:
                if sort_field.startswith('-'):
                    reverse = True
                    sort_field = sort_field[1:]  # Strip the '-'
                else:
                    reverse = False
                sort_by.append((sort_field, reverse))

            if len(sort_by) == 1 and distance_point and sort_by[0][0] == 'distance':
                sort_field, reverse = sort_by[0]
                additional_fields['_point_of_origin'] = distance_point
                lng, lat = distance_point['point'].get_coords()

                metric = xapian.GreatCircleMetric()
                center = xapian.LatLongCoord(lat, lng)

                if dwithin is not None:
                    sort_by = enquire.set_sort_by_relevance_then_key
                else:
                    sort_by = enquire.set_sort_by_key_then_relevance
                sort_by(
                    xapian.LatLongDistanceKeyMaker(self._value_column(distance_point['field']), center, metric),
                    reverse
                )
            else:
                sorter = xapian.MultiValueKeyMaker()

                for sort_field, reverse in sort_by:
                    sorter.add_value(self._value_column(sort_field), reverse)

                enquire.set_sort_by_key_then_relevance(sorter, False)

        results = []
        facets_dict = {
            'fields': {},
            'dates': {},
            'queries': {},
        }

        if not end_offset:
            end_offset = database.get_doccount() - start_offset

        matches = self._get_enquire_mset(database, enquire, start_offset, end_offset)

        for match in matches:
            app_label, module_name, pk, model_data = pickle.loads(self._get_document_data(database, match.document))
            model_data.update(additional_fields)
            if highlight:
                model_data['highlighted'] = {
                    self.content_field_name: self._do_highlight(
                        model_data.get(self.content_field_name), query
                    )
                }
            model_data['termlist'] = match.document.termlist()
            results.append(
                result_class(app_label, module_name, pk, match.percent, **model_data)
            )

        if facets:
            facets_dict['fields'] = self._do_field_facets(results, facets)
        if date_facets:
            facets_dict['dates'] = self._do_date_facets(results, date_facets)
        if query_facets:
            facets_dict['queries'] = self._do_query_facets(results, query_facets)

        return {
            'results': results,
            'hits': self._get_hit_count(database, enquire),
            'facets': facets_dict,
            'spelling_suggestion': spelling_suggestion,
        }

    def more_like_this(self, model_instance, additional_query_string=None,
                       start_offset=0, end_offset=None, models=None,
                       limit_to_registered_models=None, result_class=None, **kwargs):
        """
        Given a model instance, returns a result set of similar documents.

        Required arguments:
            `model_instance` -- The model instance to use as a basis for
                                retrieving similar documents.

        Optional arguments:
            `additional_query_string` -- An additional query string to narrow results
            `start_offset` -- The starting offset (default=0)
            `end_offset` -- The ending offset (default=None), if None, then all documents
            `limit_to_registered_models` -- Limit returned results to models registered in the current `SearchSite` (default = True)

        Returns:
            A dictionary with the following keys:
                `results` -- A list of `SearchResult`
                `hits` -- The total available results

        Opens a database connection, then builds a simple query using the
        `model_instance` to build the unique identifier.

        For each document retrieved(should always be one), adds an entry into
        an RSet (relevance set) with the document id, then, uses the RSet
        to query for an ESet (A set of terms that can be used to suggest
        expansions to the original query), omitting any document that was in
        the original query.

        Finally, processes the resulting matches and returns.
        """
        database = self._database()

        if result_class is None:
            result_class = SearchResult

        query = xapian.Query(DOCUMENT_ID_TERM_PREFIX + get_identifier(model_instance))

        enquire = xapian.Enquire(database)
        enquire.set_query(query)

        rset = xapian.RSet()

        if not end_offset:
            end_offset = database.get_doccount()

        for match in self._get_enquire_mset(database, enquire, 0, end_offset):
            rset.add_document(match.docid)

        query = xapian.Query(
            xapian.Query.OP_ELITE_SET,
            [expand.term for expand in enquire.get_eset(match.document.termlist_count(), rset, XHExpandDecider())],
            match.document.termlist_count()
        )
        query = xapian.Query(
            xapian.Query.OP_AND_NOT, [query, DOCUMENT_ID_TERM_PREFIX + get_identifier(model_instance)]
        )

        if models and len(models):
            registered_models = sorted(['%s.%s' % (model._meta.app_label, model._meta.module_name) for model in models])
        elif limit_to_registered_models:
            registered_models = self.build_models_list()
        else:
            registered_models = []

        if len(registered_models) > 0:
            query = xapian.Query(
                xapian.Query.OP_AND, query,
                xapian.Query(
                    xapian.Query.OP_OR,  [
                        xapian.Query('%s%s' % (DOCUMENT_CT_TERM_PREFIX, model)) for model in registered_models
                    ]
                )
            )

        if additional_query_string:
            query = xapian.Query(
                xapian.Query.OP_AND, query, additional_query_string
            )

        enquire.set_query(query)

        results = []
        matches = self._get_enquire_mset(database, enquire, start_offset, end_offset)

        for match in matches:
            app_label, module_name, pk, model_data = pickle.loads(self._get_document_data(database, match.document))
            results.append(
                result_class(app_label, module_name, pk, match.percent, **model_data)
            )

        return {
            'results': results,
            'hits': self._get_hit_count(database, enquire),
            'facets': {
                'fields': {},
                'dates': {},
                'queries': {},
            },
            'spelling_suggestion': None,
        }

    def parse_query(self, query_string):
        """
        Given a `query_string`, will attempt to return a xapian.Query

        Required arguments:
            ``query_string`` -- A query string to parse

        Returns a xapian.Query
        """
        if query_string == '*':
            return xapian.Query('')  # Match everything
        elif query_string == '':
            return xapian.Query()  # Match nothing

        qp = xapian.QueryParser()
        qp.set_database(self._database())
        qp.set_stemmer(xapian.Stem(self.language))
        qp.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
        qp.add_boolean_prefix('django_ct', DOCUMENT_CT_TERM_PREFIX)

        for field_dict in self.schema:
            qp.add_prefix(
                field_dict['field_name'],
                DOCUMENT_CUSTOM_TERM_PREFIX + field_dict['field_name'].upper()
            )

        vrp = XHValueRangeProcessor(self)
        qp.add_valuerangeprocessor(vrp)

        return qp.parse_query(query_string, self.flags)

    def build_schema(self, fields):
        """
        Build the schema from fields.

        Required arguments:
            ``fields`` -- A list of fields in the index

        Returns a list of fields in dictionary format ready for inclusion in
        an indexed meta-data.
        """
        content_field_name = ''
        schema_fields = [
            {'field_name': ID, 'type': 'text', 'multi_valued': 'false', 'column': 0, 'stored': True, 'mode': None},
        ]
        column = len(schema_fields)

        for field_name, field_class in sorted(fields.items(), key=lambda n: n[0]):
            if field_class.document is True:
                content_field_name = field_class.index_fieldname

            if field_class.indexed is True:
                field_data = {
                    'field_name': field_class.index_fieldname,
                    'type': 'text',
                    'multi_valued': 'false',
                    'column': column,
                    'stored': field_class.stored,
                    'mode': field_class.mode,
                }

                if field_class.field_type in ['date', 'datetime']:
                    field_data['type'] = 'date'
                elif field_class.field_type == 'integer':
                    field_data['type'] = 'long'
                elif field_class.field_type == 'float':
                    field_data['type'] = 'float'
                elif field_class.field_type == 'boolean':
                    field_data['type'] = 'boolean'
                elif field_class.field_type == 'location':
                    field_data['type'] = 'geo_point'

                if field_class.is_multivalued:
                    field_data['multi_valued'] = 'true'

                schema_fields.append(field_data)
                column += 1

        return (content_field_name, schema_fields)

    def _do_highlight(self, content, query, tag='em'):
        """
        Highlight `query` terms in `content` with html `tag`.

        This method assumes that the input text (`content`) does not contain
        any special formatting.  That is, it does not contain any html tags
        or similar markup that could be screwed up by the highlighting.

        Required arguments:
            `content` -- Content to search for instances of `text`
            `text` -- The text to be highlighted
        """
        for term in query:
            for match in re.findall('[^A-Z]+', term):  # Ignore field identifiers
                match_re = re.compile(match, re.I)
                content = match_re.sub('<%s>%s</%s>' % (tag, term, tag), content)

        return content

    def _do_field_facets(self, results, field_facets):
        """
        Private method that facets a document by field name.

        Fields of type MultiValueField will be faceted on each item in the
        (containing) list.

        Required arguments:
            `results` -- A list SearchResults to facet
            `field_facets` -- A list of fields to facet on
        """
        facet_dict = {}

        # DS_TODO: Improve this algorithm.  Currently, runs in O(N^2), ouch.
        for field in field_facets:
            facet_list = {}

            for result in results:
                field_value = getattr(result, field)
                if self._multi_value_field(field):
                    for item in field_value:  # Facet each item in a MultiValueField
                        facet_list[item] = facet_list.get(item, 0) + 1
                else:
                    facet_list[field_value] = facet_list.get(field_value, 0) + 1

            facet_dict[field] = facet_list.items()

        return facet_dict

    def _do_date_facets(self, results, date_facets):
        """
        Private method that facets a document by date ranges

        Required arguments:
            `results` -- A list SearchResults to facet
            `date_facets` -- A dictionary containing facet parameters:
                {'field': {'start_date': ..., 'end_date': ...: 'gap_by': '...', 'gap_amount': n}}
                nb., gap must be one of the following:
                    year|month|day|hour|minute|second

        For each date facet field in `date_facets`, generates a list
        of date ranges (from `start_date` to `end_date` by `gap_by`) then
        iterates through `results` and tallies the count for each date_facet.

        Returns a dictionary of date facets (fields) containing a list with
        entries for each range and a count of documents matching the range.

        eg. {
            'pub_date': [
                ('2009-01-01T00:00:00Z', 5),
                ('2009-02-01T00:00:00Z', 0),
                ('2009-03-01T00:00:00Z', 0),
                ('2009-04-01T00:00:00Z', 1),
                ('2009-05-01T00:00:00Z', 2),
            ],
        }
        """
        facet_dict = {}

        for date_facet, facet_params in date_facets.iteritems():
            gap_type = facet_params.get('gap_by')
            gap_value = facet_params.get('gap_amount', 1)
            date_range = facet_params['start_date']
            facet_list = []
            while date_range < facet_params['end_date']:
                facet_list.append((date_range.isoformat(), 0))
                if gap_type == 'year':
                    date_range = date_range.replace(
                        year=date_range.year + int(gap_value)
                    )
                elif gap_type == 'month':
                    if date_range.month + int(gap_value) > 12:
                        date_range = date_range.replace(
                            month=((date_range.month + int(gap_value)) % 12),
                            year=(date_range.year + (date_range.month + int(gap_value)) / 12)
                        )
                    else:
                        date_range = date_range.replace(
                            month=date_range.month + int(gap_value)
                        )
                elif gap_type == 'day':
                    date_range += datetime.timedelta(days=int(gap_value))
                elif gap_type == 'hour':
                    date_range += datetime.timedelta(hours=int(gap_value))
                elif gap_type == 'minute':
                    date_range += datetime.timedelta(minutes=int(gap_value))
                elif gap_type == 'second':
                    date_range += datetime.timedelta(seconds=int(gap_value))

            facet_list = sorted(facet_list, key=lambda n: n[0], reverse=True)

            for result in results:
                result_date = getattr(result, date_facet)
                if result_date:
                    if not isinstance(result_date, datetime.datetime):
                        result_date = datetime.datetime(
                            year=result_date.year,
                            month=result_date.month,
                            day=result_date.day,
                        )
                    for n, facet_date in enumerate(facet_list):
                        if result_date > datetime.datetime(*(time.strptime(facet_date[0], '%Y-%m-%dT%H:%M:%S')[0:6])):
                            facet_list[n] = (facet_list[n][0], (facet_list[n][1] + 1))
                            break

            facet_dict[date_facet] = facet_list

        return facet_dict

    def _do_query_facets(self, results, query_facets):
        """
        Private method that facets a document by query

        Required arguments:
            `results` -- A list SearchResults to facet
            `query_facets` -- A dictionary containing facet parameters:
                {'field': 'query', [...]}

        For each query in `query_facets`, generates a dictionary entry with
        the field name as the key and a tuple with the query and result count
        as the value.

        eg. {'name': ('a*', 5)}
        """
        facet_dict = {}

        for field, query in query_facets.iteritems():
            facet_dict[field] = (query, self.search(self.parse_query(query))['hits'])

        return facet_dict

    def _do_spelling_suggestion(self, database, query, spelling_query):
        """
        Private method that returns a single spelling suggestion based on
        `spelling_query` or `query`.

        Required arguments:
            `database` -- The database to check spelling against
            `query` -- The query to check
            `spelling_query` -- If not None, this will be checked instead of `query`

        Returns a string with a suggested spelling
        """
        if spelling_query:
            if ' ' in spelling_query:
                return ' '.join([database.get_spelling_suggestion(term) for term in spelling_query.split()])
            else:
                return database.get_spelling_suggestion(spelling_query)

        term_set = set()
        for term in query:
            for match in re.findall('[^A-Z]+', term):  # Ignore field identifiers
                term_set.add(database.get_spelling_suggestion(match))

        return ' '.join(term_set)

    def _database(self, writable=False):
        """
        Private method that returns a xapian.Database for use.

        Optional arguments:
            ``writable`` -- Open the database in read/write mode (default=False)

        Returns an instance of a xapian.Database or xapian.WritableDatabase
        """
        return _database(self, writable=writable)

    def _get_enquire_mset(self, database, enquire, start_offset, end_offset):
        """
        A safer version of Xapian.enquire.get_mset

        Simply wraps the Xapian version and catches any `Xapian.DatabaseModifiedError`,
        attempting a `database.reopen` as needed.

        Required arguments:
            `database` -- The database to be read
            `enquire` -- An instance of an Xapian.enquire object
            `start_offset` -- The start offset to pass to `enquire.get_mset`
            `end_offset` -- The end offset to pass to `enquire.get_mset`
        """
        try:
            return enquire.get_mset(start_offset, end_offset)
        except xapian.DatabaseModifiedError:
            # print '_get_enquire_mset.reopen()'
            database.reopen()
            return enquire.get_mset(start_offset, end_offset)

    def _get_document_data(self, database, document):
        """
        A safer version of Xapian.document.get_data

        Simply wraps the Xapian version and catches any `Xapian.DatabaseModifiedError`,
        attempting a `database.reopen` as needed.

        Required arguments:
            `database` -- The database to be read
            `document` -- An instance of an Xapian.document object
        """
        try:
            return document.get_data()
        except xapian.DatabaseModifiedError:
            # print '_get_document_data.reopen()'
            database.reopen()
            return document.get_data()

    def _get_hit_count(self, database, enquire):
        """
        Given a database and enquire instance, returns the estimated number
        of matches.

        Required arguments:
            `database` -- The database to be queried
            `enquire` -- The enquire instance
        """
        return self._get_enquire_mset(
            database, enquire, 0, database.get_doccount()
        ).size()

    def _value_column(self, field):
        """
        Private method that returns the column value slot in the database
        for a given field.

        Required arguemnts:
            `field` -- The field to lookup

        Returns an integer with the column location (0 indexed).
        """
        for field_dict in self.schema:
            if field_dict['field_name'] == field:
                return field_dict['column']
        return 0

    def _multi_value_field(self, field):
        """
        Private method that returns `True` if a field is multi-valued, else
        `False`.

        Required arguemnts:
            `field` -- The field to lookup

        Returns a boolean value indicating whether the field is multi-valued.
        """
        for field_dict in self.schema:
            if field_dict['field_name'] == field:
                return field_dict['multi_valued'] == 'true'
        return False


class XapianSearchQuery(BaseSearchQuery):
    """
    This class is the Xapian specific version of the SearchQuery class.
    It acts as an intermediary between the ``SearchQuerySet`` and the
    ``SearchBackend`` itself.
    """
    def build_params(self, *args, **kwargs):
        kwargs = super(XapianSearchQuery, self).build_params(*args, **kwargs)

        if self.end_offset is not None:
            kwargs['end_offset'] = self.end_offset - self.start_offset

        return kwargs

    def build_query(self):
        if not self.query_filter:
            query = xapian.Query('')
        else:
            query = self._query_from_search_node(self.query_filter)

        if self.models:
            subqueries = [
                xapian.Query(
                    xapian.Query.OP_SCALE_WEIGHT, xapian.Query('%s%s.%s' % (
                            DOCUMENT_CT_TERM_PREFIX,
                            model._meta.app_label, model._meta.module_name
                        )
                    ), 0  # Pure boolean sub-query
                ) for model in self.models
            ]
            query = xapian.Query(
                xapian.Query.OP_AND, query,
                xapian.Query(xapian.Query.OP_OR, subqueries)
            )

        # Removed by Kronuz in favor of boosting during _term_query() below:
        # if self.boost:
        #     print self.boost
        #     subqueries = [
        #         xapian.Query(
        #             xapian.Query.OP_SCALE_WEIGHT, self._content_field(term, False), value
        #         ) for term, value in self.boost.iteritems()
        #     ]
        #     query = xapian.Query(
        #         xapian.Query.OP_AND_MAYBE, query,
        #         xapian.Query(xapian.Query.OP_OR, subqueries)
        #     )

        return query

    def _query_from_search_node(self, search_node, is_not=False):
        query_list = []

        for child in search_node.children:
            if isinstance(child, SearchNode):
                query_list.append(
                    self._query_from_search_node(child, child.negated)
                )
            else:
                expression, term = child
                field, filter_type = search_node.split_expression(expression)

                # Handle when we've got a ``ValuesListQuerySet``...
                if hasattr(term, 'values_list'):
                    term = list(term)

                if filter_type != 'exact':
                    if isinstance(term, (list, tuple)):
                        term = [_marshal_term(t) for t in term]
                    else:
                        term = _marshal_term(term)

                if field == 'content':
                    query_list.append(self._content_field(term, is_not))
                else:
                    if filter_type == 'contains':
                        query_list.append(self._filter_contains(term, field, is_not))
                    elif filter_type == 'like':
                        query_list.append(self._filter_like(term, field, is_not))
                    elif filter_type == 'exact':
                        boost, term = (term.split('#', 1) + [None])[:2]
                        if term is not None:
                            self.add_boost(term, int(boost))
                        else:
                            term, boost = boost, None
                        query_list.append(self._filter_exact(term, field, is_not))
                    elif filter_type == 'gt':
                        query_list.append(self._filter_gt(term, field, is_not))
                    elif filter_type == 'gte':
                        query_list.append(self._filter_gte(term, field, is_not))
                    elif filter_type == 'lt':
                        query_list.append(self._filter_lt(term, field, is_not))
                    elif filter_type == 'lte':
                        query_list.append(self._filter_lte(term, field, is_not))
                    elif filter_type == 'startswith':
                        query_list.append(self._filter_startswith(term, field, is_not))
                    elif filter_type == 'in':
                        query_list.append(self._filter_in(term, field, is_not))

        if search_node.connector == 'OR':
            return xapian.Query(xapian.Query.OP_OR, query_list)
        else:
            return xapian.Query(xapian.Query.OP_AND, query_list)

    def _content_field(self, term, is_not):
        """
        Private method that returns a xapian.Query that searches for `value`
        in all fields.

        Required arguments:
            ``term`` -- The term to search for
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        if ' ' in term:
            if is_not:
                return xapian.Query(
                    xapian.Query.OP_AND_NOT, self._all_query(), self._phrase_query(
                        term.split(), self.backend.content_field_name
                    )
                )
            else:
                return self._phrase_query(term.split(), self.backend.content_field_name)
        else:
            if is_not:
                return xapian.Query(xapian.Query.OP_AND_NOT, self._all_query(), self._term_query(term))
            else:
                return self._term_query(term)

    def _filter_contains(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for `term`
        in a specified `field`.

        Required arguments:
            ``term`` -- The term to search for
            ``field`` -- The field to search
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        if ' ' in term:
            return self._filter_exact(term, field, is_not)
        else:
            if is_not:
                return xapian.Query(xapian.Query.OP_AND_NOT, self._all_query(), self._term_query(term, field))
            else:
                return self._term_query(term, field)

    def _filter_exact(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for `term`
        in a specified `field`.

        Required arguments:
            ``term`` -- The term to search for
            ``field`` -- The field to search
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        if is_not:
            return xapian.Query(xapian.Query.OP_AND_NOT, self._all_query(), self._term_query_exact(term, field))
        else:
            return self._term_query_exact(term, field)

    def _filter_like(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for `term`
        in a specified `field`.

        Required arguments:
            ``term`` -- The term to search for
            ``field`` -- The field to search
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        if ' ' in term:
            if is_not:
                return xapian.Query(
                    xapian.Query.OP_AND_NOT, self._all_query(), self._phrase_query(term.split(), field)
                )
            else:
                return self._phrase_query(term.split(), field)
        else:
            if is_not:
                return xapian.Query(xapian.Query.OP_AND_NOT, self._all_query(), self._term_query(term, field))
            else:
                return self._term_query(term, field)

    def _filter_in(self, term_list, field, is_not):
        """
        Private method that returns a xapian.Query that searches for any term
        of `value_list` in a specified `field`.

        Required arguments:
            ``term_list`` -- The terms to search for
            ``field`` -- The field to search
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        query_list = []
        for term in term_list:
            if ' ' in term:
                query_list.append(
                    self._phrase_query(term.split(), field)
                )
            else:
                query_list.append(
                    self._term_query(term, field)
                )
        if is_not:
            return xapian.Query(xapian.Query.OP_AND_NOT, self._all_query(), xapian.Query(xapian.Query.OP_OR, query_list))
        else:
            return xapian.Query(xapian.Query.OP_OR, query_list)

    def _filter_startswith(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for any term
        that begins with `term` in a specified `field`.

        Required arguments:
            ``term`` -- The terms to search for
            ``field`` -- The field to search
            ``is_not`` -- Invert the search results

        Returns:
            A xapian.Query
        """
        if is_not:
            return xapian.Query(
                xapian.Query.OP_AND_NOT,
                self._all_query(),
                self.backend.parse_query('%s:%s*' % (field, term)),
            )
        return self.backend.parse_query('%s:%s*' % (field, term))

    def _filter_gt(self, term, field, is_not):
        return self._filter_lte(term, field, is_not=(is_not != True))

    def _filter_lt(self, term, field, is_not):
        return self._filter_gte(term, field, is_not=(is_not != True))

    def _filter_gte(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for any term
        that is greater than `term` in a specified `field`.
        """
        vrp = XHValueRangeProcessor(self.backend)
        pos, begin, end = vrp('%s:%s' % (field, _marshal_value(term)), '*')
        if is_not:
            return xapian.Query(xapian.Query.OP_AND_NOT,
                self._all_query(),
                xapian.Query(xapian.Query.OP_VALUE_RANGE, pos, begin, end)
            )
        return xapian.Query(xapian.Query.OP_VALUE_RANGE, pos, begin, end)

    def _filter_lte(self, term, field, is_not):
        """
        Private method that returns a xapian.Query that searches for any term
        that is less than `term` in a specified `field`.
        """
        vrp = XHValueRangeProcessor(self.backend)
        pos, begin, end = vrp('%s:' % field, '%s' % _marshal_value(term))
        if is_not:
            return xapian.Query(xapian.Query.OP_AND_NOT,
                self._all_query(),
                xapian.Query(xapian.Query.OP_VALUE_RANGE, pos, begin, end)
            )
        return xapian.Query(xapian.Query.OP_VALUE_RANGE, pos, begin, end)

    def _all_query(self):
        """
        Private method that returns a xapian.Query that returns all documents,

        Returns:
            A xapian.Query
        """
        return xapian.Query('')

    def _term_query(self, term, field=None):
        """
        Private method that returns a term based xapian.Query that searches
        for `term`.

        Required arguments:
            ``term`` -- The term to search for
            ``field`` -- The field to search (If `None`, all fields)

        Returns:
            A xapian.Query
        """
        stem = xapian.Stem(self.backend.language)

        if field == 'id':
            query = xapian.Query('%s%s' % (DOCUMENT_ID_TERM_PREFIX, term))
        elif field == 'django_ct':
            query = xapian.Query('%s%s' % (DOCUMENT_CT_TERM_PREFIX, term))
        else:
            if field:
                stemmed = 'Z%s%s%s' % (
                    DOCUMENT_CUSTOM_TERM_PREFIX, field.upper(), stem(term)
                )
                unstemmed = '%s%s%s' % (
                    DOCUMENT_CUSTOM_TERM_PREFIX, field.upper(), term
                )
            else:
                stemmed = 'Z%s' % stem(term)
                unstemmed = term
            query = xapian.Query(
                xapian.Query.OP_OR,
                xapian.Query(stemmed),
                xapian.Query(unstemmed)
            )
        if term in self.boost:
            query = xapian.Query(
                xapian.Query.OP_FILTER,
                xapian.Query(xapian.FixedWeightPostingSource(self.boost[term])),
                query
            )
        return query

    def _term_query_exact(self, term, field=None):
        """
        Private method that returns a term based xapian.Query that searches
        for an exact `term`.
        Required arguments:
            ``term`` -- The term to search for
            ``field`` -- The field to search (If `None`, all fields)
        Returns:
            A xapian.Query
        """
        if field:
            query = xapian.Query('%s%s%s' % (
                    DOCUMENT_CUSTOM_TERM_PREFIX, field.upper(), term
                )
            )
        else:
            query = xapian.Query(term)
        if term in self.boost:
            query = xapian.Query(
                xapian.Query.OP_FILTER,
                xapian.Query(xapian.FixedWeightPostingSource(self.boost[term])),
                query
            )
        return query

    def _phrase_query(self, term_list, field=None):
        """
        Private method that returns a phrase based xapian.Query that searches
        for terms in `term_list.

        Required arguments:
            ``term_list`` -- The terms to search for
            ``field`` -- The field to search (If `None`, all fields)

        Returns:
            A xapian.Query
        """
        if field:
            return xapian.Query(
                xapian.Query.OP_PHRASE, [
                    '%s%s%s' % (
                        DOCUMENT_CUSTOM_TERM_PREFIX, field.upper(), term
                    ) for term in term_list
                ]
            )
        else:
            return xapian.Query(xapian.Query.OP_PHRASE, term_list)


def _marshal_value(value):
    """
    Private utility method that converts Python values to a string for Xapian values.
    """
    if isinstance(value, datetime.datetime):
        value = _marshal_datetime(value)
    elif isinstance(value, datetime.date):
        value = _marshal_date(value)
    elif isinstance(value, bool):
        if value:
            value = u't'
        else:
            value = u'f'
    elif isinstance(value, float):
        value = xapian.sortable_serialise(value)
    elif isinstance(value, (int, long)):
        value = u'%012d' % value
    else:
        value = normalize(force_unicode(value))
    return value


def _marshal_term(term):
    """
    Private utility method that converts Python terms to a string for Xapian terms.
    """
    if isinstance(term, datetime.datetime):
        term = _marshal_datetime(term)
    elif isinstance(term, datetime.date):
        term = _marshal_date(term)
    elif isinstance(term, bool):
        if term:
            term = u't'
        else:
            term = u'f'
    elif isinstance(term, float):
        term = u'%g' % term
    elif isinstance(term, (int, long)):
        term = u'%d' % term
    else:
        term = normalize(force_unicode(term))
    return term


def _marshal_tagged_terms(terms):
    if isinstance(terms, dict):
        _terms = {}
        for term, weight in terms.items():
            term = term.lower()
            _terms[term] = weight
        terms = _terms
    else:
        terms = {terms.lower(): 1}
    for term, weight in terms.items():
        term = term.rsplit('=', 1)[0]
        terms[term] = max(terms.get(term, 0), int(weight * 0.5))
        #term = term.split(':', 1)[0]
        #terms[term] = max(terms.get(term, 0), int(weight * 0.2))
    return terms


def _marshal_weighted_terms(terms, minlen=0, minper=0.3):
    split_terms = {}
    if not isinstance(terms, dict):
        terms = {terms: 1}
    for term, weight in terms.items():
        for _term in _marshal_term(term).split():
            split_terms[_term] = max(split_terms.get(_term, 0), weight)
    # Find all the substrings of the term (all digit terms treated differently):
    final_terms = {}
    for term, weight in split_terms.items():
        for i in range(len(term)):
            for j in range(i + 1, len(term) + 1):
                if j - i > minlen and j - i >= int(len(term) * minper):
                    _term = term[i:j]
                    _weight = int(float(weight * (j - i)) / len(term))
                    final_terms[_term] = max(final_terms.get(_term, 0), _weight)
    return final_terms


def _marshal_date(d):
    return u'%04d%02d%02d000000' % (d.year, d.month, d.day)


def _marshal_datetime(dt):
    if dt.microsecond:
        return u'%04d%02d%02d%02d%02d%02d%06d' % (
            dt.year, dt.month, dt.day, dt.hour,
            dt.minute, dt.second, dt.microsecond
        )
    else:
        return u'%04d%02d%02d%02d%02d%02d' % (
            dt.year, dt.month, dt.day, dt.hour,
            dt.minute, dt.second
        )


class XapianEngine(BaseEngine):
    backend = XapianSearchBackend
    query = XapianSearchQuery
