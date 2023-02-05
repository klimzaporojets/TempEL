import logging
import re
import xml.sax
from datetime import datetime
from typing import List

from src.utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


class WikipediaHistoryReader(xml.sax.ContentHandler):
    # def __init__(self, filter_namespace, filters_date, article_callback, v_nr_pages_with_change_of_title,
    #              v_nr_pages_change_title_error,
    #              v_nr_scanned_entities,
    #              max_look_back_for_stable_page_version,
    #              min_days_stable_page_version, do_asserts=False, filter_pages=None):
    def __init__(self, filter_namespace, filters_date, article_callback,
                 v_nr_pages_change_title_error,
                 max_look_back_for_stable_page_version,
                 min_days_stable_page_version, do_asserts=False, filter_pages=None):
        super().__init__()
        self.v_nr_pages_change_title_error = v_nr_pages_change_title_error
        logger.debug('Init WikipediaHistoryReader')
        self.do_asserts = do_asserts
        self.stack_elements: List = list()
        self.max_look_back_for_stable_page_version = max_look_back_for_stable_page_version
        self.min_days_stable_page_version = min_days_stable_page_version

        self.field_revision_text = dict()
        self.prev_revision_date = dict()
        self.prev_revision_date_str = dict()
        self.prev_revision_text = dict()
        self.secured_content_text = dict()
        self.secured_revision_date = dict()
        self.field_revision_date = dict()
        self.max_time_lapse_between_revisions = dict()

        self.filter_pages = filter_pages
        self.should_be_processed = True
        self.nr_revisions = 0

        self.filter_max_date = None
        max_date_of_filter = None
        for curr_date_str, curr_date_filter in filters_date.items():
            curr_date = datetime.strptime(curr_date_str, '%Y-%m-%dT%H:%M:%SZ')
            if max_date_of_filter is None:
                max_date_of_filter = curr_date
                self.filter_max_date = curr_date_filter
            else:
                if curr_date > max_date_of_filter:
                    self.filter_max_date = curr_date_filter

            self.field_revision_text[curr_date_str] = None
            self.prev_revision_date[curr_date_str] = None
            self.prev_revision_date_str[curr_date_str] = None
            self.prev_revision_text[curr_date_str] = None
            self.field_revision_date[curr_date_str] = None
            self.secured_content_text[curr_date_str] = None
            self.secured_revision_date[curr_date_str] = None
            self.max_time_lapse_between_revisions[curr_date_str] = 0

        self.text = ''
        self.field_title = ''
        self.field_page_id = ''
        self.field_comment = ''
        self.redirTarget = None
        self.ns = ''

        self.min_date_to_filter = None
        # max date in all the filters --> can be used to quickly filter in content for instance
        self.max_date_to_filter = None

        self.timestamp = ''
        self.revision_date = None
        self.revision_date_str = None

        self.creation_date = None
        self.field_creation_date = None

        self.num_articles = 0
        self.tick = 0

        self.filter_namespace = filter_namespace
        self.filters_date = filters_date

        self.filters_date_parsed_date = dict()
        for curr_date_str in self.filters_date.keys():
            self.filters_date_parsed_date[curr_date_str] = datetime.strptime(curr_date_str, '%Y-%m-%dT%H:%M:%SZ')

        self.article_callback = article_callback
        self.tot_size = 0
        self.processed_size = 0
        self.prev_processed_size = 0

        # in true when we are inside the elements, the idea is to avoid doing "in self.stack_elements" operation
        # which is O(n)
        self.active_page = False
        self.active_revision = False

        # related to movement of titles
        self.page_change_of_titles = list()
        self.nr_processed_revisions = 0
        self.old_page_title = None
        self.new_page_title = None

        self.move_patterns = ['(^|\s)moved (\[\[.*?\]\]) to (\[\[.*?\]\])',
                              '(^|\s)moved page (\[\[.*?\]\]) to (\[\[.*?\]\])']

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startElementNS(self, name, qname, attrs):
        pass

    def startElement(self, name, attributes: xml.sax.xmlreader.AttributesImpl):

        if name == 'ns':
            if self.do_asserts:
                assert self.stack_elements == ['page']
        elif name == 'page':
            self.should_be_processed = True
            if self.do_asserts:
                assert self.stack_elements == []
            self.nr_revisions = 0
            self.field_title = ''
            self.redirTarget = None
            self.active_page = True
            self.revision_date = None
            self.creation_date = None
            self.field_creation_date = None
            self.revision_date_str = None
            self.nr_processed_revisions = 0
            self.page_change_of_titles = list()

            for curr_date_str in self.filters_date.keys():
                self.field_revision_text[curr_date_str] = None
                self.prev_revision_date[curr_date_str] = None
                self.prev_revision_date_str[curr_date_str] = None
                self.prev_revision_text[curr_date_str] = None
                self.secured_content_text[curr_date_str] = None
                self.secured_revision_date[curr_date_str] = None
                self.field_revision_date[curr_date_str] = None
                self.max_time_lapse_between_revisions[curr_date_str] = 0

            self.ns = ''

        elif name == 'revision':
            if self.do_asserts:
                assert self.stack_elements == ['page']

            if self.filter_pages is not None and self.nr_revisions == 0:
                if self.filter_pages(self.field_page_id):
                    logger.info('GOOD! FOUND THE FOLLOWING: {} - {}'.format(self.field_title, self.field_page_id))
                else:
                    self.should_be_processed = False
            self.nr_revisions += 1
            self.text = ''
            self.redirTarget = None
            self.active_revision = True
            self.revision_date = None
            # in case the revision is a change of title of the page
            self.old_page_title = None
            self.new_page_title = None

        elif name == 'title':
            if self.do_asserts:
                assert self.stack_elements == ['page']
                assert self.field_title == ''
            self.field_title = ''
        elif name == 'text':
            if self.do_asserts:
                assert self.stack_elements == ['page', 'revision']
            if self.redirTarget is not None:
                return
            self.text = ''
        elif name == 'id':
            if self.active_page and not self.active_revision:
                self.field_page_id = ''
        elif name == 'timestamp':
            if self.active_page and self.active_revision:
                self.timestamp = ''
        elif name == 'comment':
            self.field_comment = ''
        else:
            if self.do_asserts:
                assert len(self.stack_elements) == 0 or self.stack_elements[-1] == 'page' or \
                       self.stack_elements[-1] == 'revision'
            return

        self.stack_elements.append(name)

    def endElement(self, name):

        if len(self.stack_elements) > 0 and name == self.stack_elements[-1]:
            self.stack_elements.pop()
        if not self.should_be_processed:
            if name == 'revision':
                self.active_revision = False
            if name == 'page':
                self.active_page = False
            return
        if name == 'comment':
            self.field_comment = self.field_comment.strip()
            # the first revision can not just consist of moving the current page, sometimes there is a "move" comment
            # but referring to another page
            if self.nr_processed_revisions > 0:
                for curr_move_pattern in self.move_patterns:
                    result_pattern = re.search(curr_move_pattern, self.field_comment)
                    if result_pattern is not None:
                        self.old_page_title = result_pattern.group(2)[2:-2].strip()
                        self.new_page_title = result_pattern.group(3)[2:-2].strip()
                        if not (']]' in self.old_page_title or ']]' in self.new_page_title):
                            break  # once a pattern is found, no need to look for other patterns
                        else:
                            self.old_page_title = None
                            self.new_page_title = None

        if name == 'timestamp' and self.active_revision and self.active_page:
            if self.filter_namespace(self.ns):
                self.revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                self.revision_date_str = self.timestamp
                # if it is the first revision, then assigns the creation date
                if self.creation_date is None or self.revision_date < self.creation_date:
                    self.creation_date = self.revision_date
                    self.field_creation_date = self.timestamp

        if name == 'revision':
            # 20/03/2022 - added the control of self.text.strip() in order to avoid adding empty text
            if self.filter_namespace(self.ns) and self.text.strip() != '':
                assert self.revision_date is not None
                for curr_filtered_date, curr_filter_date in self.filters_date.items():
                    # we are interested in getting the latest revision as long as the filter is ok with this
                    if (curr_filter_date(self.revision_date) or
                        (self.prev_revision_date[curr_filtered_date] is not None and
                         curr_filter_date(self.prev_revision_date[curr_filtered_date]))) \
                            and \
                            ((self.prev_revision_date[curr_filtered_date] is None) or
                             (self.revision_date > self.prev_revision_date[curr_filtered_date])):

                        if self.prev_revision_date[curr_filtered_date] is None:
                            # IF it is the first one, puts it anyway, no other choice
                            self.prev_revision_text[curr_filtered_date] = self.text
                            self.field_revision_text[curr_filtered_date] = self.text
                            self.field_revision_date[curr_filtered_date] = self.revision_date_str

                            self.prev_revision_date[curr_filtered_date] = self.revision_date
                            self.prev_revision_date_str[curr_filtered_date] = self.revision_date_str
                            self.secured_content_text[curr_filtered_date] = self.text
                            self.secured_revision_date[curr_filtered_date] = self.revision_date_str
                        else:
                            curr_filtered_date_parsed = self.filters_date_parsed_date[curr_filtered_date]

                            curr_lapse_from_filter_date = curr_filtered_date_parsed - self.revision_date

                            curr_lapse_from_filter_date = curr_lapse_from_filter_date.days
                            if curr_lapse_from_filter_date > self.max_look_back_for_stable_page_version:
                                if curr_filter_date(self.revision_date):
                                    self.secured_content_text[curr_filtered_date] = self.text
                                    self.secured_revision_date[curr_filtered_date] = self.revision_date_str
                                else:
                                    assert self.prev_revision_date_str[curr_filtered_date] is not None
                                    assert curr_filter_date(self.prev_revision_date[curr_filtered_date])
                                    self.secured_content_text[curr_filtered_date] = self.prev_revision_text[
                                        curr_filtered_date]
                                    self.secured_revision_date[curr_filtered_date] = self.prev_revision_date_str[
                                        curr_filtered_date]

                                self.prev_revision_text[curr_filtered_date] = self.text
                                self.field_revision_text[curr_filtered_date] = self.text
                                self.field_revision_date[curr_filtered_date] = self.revision_date_str
                                self.prev_revision_date[curr_filtered_date] = self.revision_date
                                self.prev_revision_date_str[curr_filtered_date] = self.revision_date_str
                            else:
                                # here control well in order to not assign malicious content
                                curr_lapse = self.revision_date - self.prev_revision_date[curr_filtered_date]
                                curr_lapse = curr_lapse.total_seconds()
                                assert curr_lapse >= 0
                                assert curr_filter_date(self.prev_revision_date[curr_filtered_date])
                                if curr_lapse > self.max_time_lapse_between_revisions[curr_filtered_date] or \
                                        (curr_lapse / 86400) >= self.min_days_stable_page_version:
                                    self.secured_content_text[curr_filtered_date] = \
                                        self.prev_revision_text[curr_filtered_date]

                                    self.secured_revision_date[curr_filtered_date] = \
                                        self.prev_revision_date_str[curr_filtered_date]
                                    self.max_time_lapse_between_revisions[curr_filtered_date] = \
                                        max(curr_lapse, self.max_time_lapse_between_revisions[curr_filtered_date])

                                self.prev_revision_text[curr_filtered_date] = self.text
                                self.field_revision_text[curr_filtered_date] = self.text
                                self.field_revision_date[curr_filtered_date] = self.revision_date_str
                                self.prev_revision_date[curr_filtered_date] = self.revision_date
                                self.prev_revision_date_str[curr_filtered_date] = self.revision_date_str

                if self.old_page_title is not None and self.new_page_title is not None:
                    self.page_change_of_titles.append({'old_page_title': self.old_page_title,
                                                       'new_page_title': self.new_page_title,
                                                       'page_id': self.field_page_id.strip(),
                                                       'date': self.revision_date})

            self.active_revision = False
            self.nr_processed_revisions += 1

        if name == 'page':

            self.active_page = False
            titles_of_page_per_date = dict()

            if self.filter_namespace(self.ns):
                ignore_page = False
                len_page_change_titles = len(self.page_change_of_titles)
                self.field_title = self.field_title.strip()

                for curr_filtered_date, curr_filter_date in self.filters_date.items():
                    if self.field_title != '' and self.field_revision_text[curr_filtered_date] is not None:
                        if len_page_change_titles > 0:

                            change_title_idx = 0
                            # last_changed_idx_filtered_date = -1
                            while change_title_idx < len_page_change_titles and \
                                    curr_filter_date(self.page_change_of_titles[change_title_idx]['date']):
                                change_title_idx += 1
                            change_title_idx -= 1
                            change_title_idx = max(0, change_title_idx)

                            change_title_entry = self.page_change_of_titles[change_title_idx]

                            # if it is the last, asserts that it coincides with self.field_title
                            if change_title_idx + 1 == len_page_change_titles:
                                if change_title_entry['new_page_title'] != self.field_title:
                                    logger.warning('WARNING error in assertion of new_page_title for page "%s" '
                                                   'page_id: %s new_page_title is "%s"' %
                                                   (self.field_title, self.field_page_id,
                                                    change_title_entry['new_page_title']))

                                    logger.warning('WARNING error all the change_title_entry is: %s' %
                                                   change_title_entry)
                                    with self.v_nr_pages_change_title_error.get_lock():
                                        self.v_nr_pages_change_title_error.value += 1
                                    ignore_page = False

                                    change_title_entry['new_page_title'] = self.field_title

                            if not curr_filter_date(change_title_entry['date']):
                                # if the filter date happens before the actual change in title, then the
                                # title of the page before is put
                                titles_of_page_per_date[curr_filtered_date] = {
                                    'title': change_title_entry['old_page_title'],
                                    'current_title': self.field_title,
                                    'filtered_date': curr_filtered_date,
                                    'page_id': self.field_page_id}
                            else:
                                titles_of_page_per_date[curr_filtered_date] = {
                                    'title': change_title_entry['new_page_title'],
                                    'current_title': self.field_title,
                                    'filtered_date': curr_filtered_date,
                                    'page_id': self.field_page_id}
                        else:
                            titles_of_page_per_date[curr_filtered_date] = {'title': self.field_title,
                                                                           'current_title': self.field_title,
                                                                           'filtered_date': curr_filtered_date,
                                                                           'page_id': self.field_page_id}

                    if self.prev_revision_date[curr_filtered_date] is not None:
                        curr_filtered_date_parsed = self.filters_date_parsed_date[curr_filtered_date]
                        lapse_between_last_version_and_cut = curr_filtered_date_parsed - \
                                                             self.prev_revision_date[curr_filtered_date]
                        lapse_between_last_version_and_cut = lapse_between_last_version_and_cut.total_seconds()

                        mtb_rev = self.max_time_lapse_between_revisions[curr_filtered_date]
                        if mtb_rev > 0.0 and \
                                ((mtb_rev <= lapse_between_last_version_and_cut) or
                                 (lapse_between_last_version_and_cut / 86400) >= self.min_days_stable_page_version):
                            self.secured_revision_date[curr_filtered_date] = \
                                self.field_revision_date[curr_filtered_date]
                            self.secured_content_text[curr_filtered_date] = \
                                self.field_revision_text[curr_filtered_date]

                if not ignore_page:
                    self.article_callback(self.field_title,
                                          self.secured_content_text,  # dict entry by curr_filtered_date
                                          self.field_page_id,
                                          self.field_creation_date,
                                          self.secured_revision_date,  # dict entry by curr_filtered_date
                                          titles_of_page_per_date)

    def endElementNS(self, name, qname):
        pass

    def characters(self, content):
        if self.nr_revisions >= 1 and not self.should_be_processed:
            return

        if len(self.stack_elements) == 0:
            return

        assert content is not None
        stack_min_1 = self.stack_elements[-1]
        if stack_min_1 == 'ns':
            self.ns += content
            logger.debug('Set ns to {0}'.format(self.ns))

        if self.ns != '' and not self.filter_namespace(self.ns):
            return
        if self.revision_date is not None and not self.filter_max_date(self.revision_date):
            return

        if stack_min_1 == 'title':
            self.field_title += content
            logger.debug('Set title to {0}'.format(self.field_title))

        if stack_min_1 == 'text':
            self.text += content

        if stack_min_1 == 'id':
            if self.active_page and not self.active_revision:
                self.field_page_id += content

        if stack_min_1 == 'timestamp':
            if self.active_page and self.active_revision:
                self.timestamp += content

        if stack_min_1 == 'comment':
            if self.active_page and self.active_revision:
                self.field_comment += content

    def startPrefixMapping(self, prefix, uri):
        pass

    def endPrefixMapping(self, prefix):
        pass
