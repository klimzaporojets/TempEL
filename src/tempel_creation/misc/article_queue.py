import logging
import multiprocessing

from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


class ArticleReadingQueue():
    def __init__(self, article_queue_size=200,
                 link_stats_queue_size=200, page_info_stats_queue_size=200,
                 title_changes_queue_size=200, evolution_content_queue_size=200,
                 process_file_queue_size=20000):
        manager = multiprocessing.Manager()
        self.manager = manager
        self.article_queue = manager.Queue(maxsize=article_queue_size)
        self.mentions_queue = manager.Queue(maxsize=200)
        self.wikidata_entity_queue = manager.Queue(maxsize=200)
        self.page_info_stats_queue = manager.Queue(maxsize=page_info_stats_queue_size)
        self.entity_info_stats_queue = manager.Queue(maxsize=200)
        self.link_stats_queue = manager.Queue(maxsize=link_stats_queue_size)
        self.title_changes_queue = manager.Queue(maxsize=title_changes_queue_size)
        self.qid_changes_queue = manager.Queue(maxsize=200)
        self.evolution_content_queue = manager.Queue(maxsize=evolution_content_queue_size)

        # with file names to process
        # process_files_queue has to be big to fit all the files (see wikipedia_create_dataset.py).
        self.process_files_queue = manager.Queue(maxsize=process_file_queue_size)
        self.gatherer_encodings_queue = manager.Queue(maxsize=200)
        self.gatherer_encodings_queue2 = manager.Queue(maxsize=200)
        self.gatherer_top_changes_queue = manager.Queue(maxsize=200)

    def enqueue_article(self, title, source, page_id, creation_date, revision_date, title_of_page_info):
        """
        :param title:
        :param source:
        :param page_id:
        :param creation_date:
        :param revision_date:
        :param filtered_date:
        :return:
        """
        logger.debug("Enqueue article {0}".format(title))
        self.article_queue.put(
            (title, source, page_id, creation_date, revision_date, title_of_page_info))
        logger.debug("Done")

    def enqueue_article_wikidata(self, internal_id, title_qid, creation_date, last_revision_date, source):
        self.wikidata_entity_queue.put((internal_id, title_qid, creation_date, last_revision_date, source))

    def enqueue_page_info_stats_writer(self, page_info_json):
        self.page_info_stats_queue.put(page_info_json)

    def enqueue_link_stats_writer(self, page_link_stats_json):
        self.link_stats_queue.put(page_link_stats_json)
