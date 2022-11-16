import gzip
import logging
import os
import pickle
import traceback

from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                      path_wikipedia_wikidata_map):
    nr_pages_processed = 0
    wikipedia_page_id_to_wikidata_qid = dict()
    if os.path.exists(path_cache_wikipedia_page_id_to_wikidata_qid):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_id_to_wikidata_qid)
        wikipedia_page_id_to_wikidata_qid = pickle.load(open(path_cache_wikipedia_page_id_to_wikidata_qid, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_wikidata_qid)
    else:
        logger.info('starting processing the following file in load_wiki_page_id_to_wikidata_qid: %s' %
                    path_wikipedia_wikidata_map)
        with gzip.open(path_wikipedia_wikidata_map, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('ISO-8859-1')
                insert_start_with = 'INSERT INTO `page_props` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ',\''
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            inter_p = '\',\''
                            index_f = rest.index(inter_p)
                            field_property = rest[:index_f].strip()
                            rest = rest[index_f + len(inter_p):]
                            index_f = rest.index('\',')
                            field_value = rest[:index_f].strip()
                            if field_property == 'wikibase_item':
                                assert field_page_id_from not in wikipedia_page_id_to_wikidata_qid
                                wikipedia_page_id_to_wikidata_qid[field_page_id_from] = field_value
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('nr of processed pages: %s' % nr_pages_processed)
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_wikidata_qid some sort of error with %s ' %
                                         curr_insert_tuple)
                            logger.error(err)
                        finally:
                            pass
        pickle.dump(wikipedia_page_id_to_wikidata_qid, open(path_cache_wikipedia_page_id_to_wikidata_qid, 'wb'))

    return wikipedia_page_id_to_wikidata_qid


# def load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
#                                             wikipedia_page_title_to_wikipedia_page_id,
#                                             path_wikipedia_page_redirects,
#                                             output_dir):
def load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                            wikipedia_page_title_to_wikipedia_page_id,
                                            path_wikipedia_page_redirects):
    nr_pages_processed = 0
    wikipedia_page_id_to_redirected_page_id = dict()
    if os.path.exists(path_cache_wikipedia_page_id_to_redirected_page_id):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_id_to_redirected_page_id)
        wikipedia_page_id_to_redirected_page_id = \
            pickle.load(open(path_cache_wikipedia_page_id_to_redirected_page_id, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_redirected_page_id)
    else:
        logger.info('starting procesisng the following file in load_wiki_page_id_to_redirected_page_id: %s' %
                    path_wikipedia_page_redirects)

        # just for logs, for now None
        file_out = None
        # file_out = open(os.path.join(output_dir, 'output_page_id_to_redirected_page_id.log'), 'wt', encoding='utf-8')
        with gzip.open(path_wikipedia_page_redirects, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8')
                insert_start_with = 'INSERT INTO `redirect` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ','
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            # -----
                            inter_p = ',\''
                            index_f = rest.index(inter_p)
                            field_namespace = int(rest[:index_f].strip())
                            rest = rest[index_f + len(inter_p):]
                            # -----
                            inter_p = '\',\''
                            index_f = rest.index(inter_p)
                            field_title_to = rest[:index_f].strip()
                            if '\'' in field_title_to:
                                field_title_to = field_title_to.replace('\\\'', "'")

                            if field_namespace == 0:
                                page_id_to = wikipedia_page_title_to_wikipedia_page_id[field_title_to]
                                if file_out is not None:
                                    file_out.write(str(field_page_id_from) + ' --- ' + str(field_title_to) + '\n')
                                    file_out.flush()
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('have processed to get page id to title redirections: %s'
                                                % nr_pages_processed)
                                assert field_page_id_from not in wikipedia_page_id_to_redirected_page_id
                                wikipedia_page_id_to_redirected_page_id[field_page_id_from] = page_id_to
                        except Exception as err:
                            logger.error('!!!load_wiki_page_id_to_redirected_page_id some sort of error when '
                                         'processing redirects with %s' % curr_insert_tuple)
                            logger.error(traceback.format_exc())
                        finally:
                            pass
        pickle.dump(wikipedia_page_id_to_redirected_page_id,
                    open(path_cache_wikipedia_page_id_to_redirected_page_id, 'wb'))

    return wikipedia_page_id_to_redirected_page_id


def load_wiki_page_title_to_wiki_page_id(path_cache_wikipedia_page_title_to_wikipedia_page_id,
                                         path_cache_wikipedia_page_id_to_wikipedia_page_title,
                                         path_wikipedia_page_info):
    nr_pages_processed = 0
    wikipedia_page_title_to_wikipedia_page_id = dict()
    wikipedia_page_id_to_wikipedia_page_title = dict()
    if os.path.exists(path_cache_wikipedia_page_title_to_wikipedia_page_id):
        logger.info('starting loading from pickle %s' % path_cache_wikipedia_page_title_to_wikipedia_page_id)
        wikipedia_page_title_to_wikipedia_page_id = \
            pickle.load(open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'rb'))
        wikipedia_page_id_to_wikipedia_page_title = \
            pickle.load(open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'rb'))
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_title_to_wikipedia_page_id)
        logger.info('loaded from pickle %s' % path_cache_wikipedia_page_id_to_wikipedia_page_title)
    else:
        logger.info('starting loading in load_wiki_page_title_to_wiki_page_id from %s' % path_wikipedia_page_info)

        # just for logs, for now None
        file_out = None
        with gzip.open(path_wikipedia_page_info, 'r') as f:
            for idx_line, line in enumerate(f):
                line = line.decode('utf-8')
                insert_start_with = 'INSERT INTO `page` VALUES'
                if line.startswith(insert_start_with):
                    str_inserts = line[len(insert_start_with):]
                    splitted_inserts = str_inserts.split('),(')
                    splitted_inserts[0] = splitted_inserts[0].strip()[1:]
                    splitted_inserts[-1] = splitted_inserts[-1].strip()[:-1]
                    for curr_insert_tuple in splitted_inserts:
                        try:
                            inter_p = ','
                            index_f = curr_insert_tuple.index(inter_p)
                            field_page_id_from = curr_insert_tuple[:index_f].strip()
                            field_page_id_from = int(field_page_id_from)
                            rest = curr_insert_tuple[index_f + len(inter_p):]
                            # -----
                            inter_p = ',\''
                            index_f = rest.index(inter_p)
                            field_namespace = int(rest[:index_f].strip())
                            rest = rest[index_f + len(inter_p):]
                            # -----
                            inter_p = '\',\''
                            index_f = rest.index(inter_p)
                            field_title_to = rest[:index_f].strip()

                            if field_namespace == 0:
                                if file_out is not None:
                                    file_out.write(str(field_page_id_from) + ' --- ' + str(field_title_to) + '\n')
                                    file_out.flush()
                                nr_pages_processed += 1
                                if nr_pages_processed % 1000000 == 0:
                                    logger.info('have processed to get page id to title mapping: %s' %
                                                nr_pages_processed)

                                if '\'' in field_title_to:
                                    field_title_to = field_title_to.replace('\\\'', "'")

                                assert field_title_to not in wikipedia_page_title_to_wikipedia_page_id

                                wikipedia_page_title_to_wikipedia_page_id[field_title_to] = field_page_id_from
                                assert field_page_id_from not in wikipedia_page_id_to_wikipedia_page_title
                                wikipedia_page_id_to_wikipedia_page_title[field_page_id_from] = field_title_to

                        except Exception as err:
                            logger.error('!!!load_wiki_page_title_to_wiki_page_id - some sort of error with %s' %
                                         curr_insert_tuple)
                            logger.error(traceback.format_exc())
                        finally:
                            pass
        if file_out is not None:
            file_out.flush()
            file_out.close()
        pickle.dump(wikipedia_page_title_to_wikipedia_page_id,
                    open(path_cache_wikipedia_page_title_to_wikipedia_page_id, 'wb'))
        pickle.dump(wikipedia_page_id_to_wikipedia_page_title,
                    open(path_cache_wikipedia_page_id_to_wikipedia_page_title, 'wb'))

    return wikipedia_page_title_to_wikipedia_page_id, wikipedia_page_id_to_wikipedia_page_title
