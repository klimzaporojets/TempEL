import logging
import re
import traceback
from multiprocessing.managers import DictProxy
from time import sleep

import mwparserfromhell
from mwparserfromhell.wikicode import Wikicode

from tempel_creation.misc.recursive_cleaning import recursive_clean, recursive_clean_klim
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)


def expand_convert_api(convert_text, request_session):
    URL = 'https://en.wikipedia.org/w/api.php'

    PARAMS = {
        'action': 'expandtemplates',
        # "text": "'is [[Karman Line|{{convert|100|km|mi|abbr=on}}]] a weapon? '",
        # "text": "toilets or bathrooms. SRO units range from {{convert|80|to|140|ft2|m2|order=flip|0}}.",
        'text': convert_text,
        'prop': 'wikitext',
        'format': 'json'
    }
    # with v_lock:
    R = request_session.get(url=URL, params=PARAMS, timeout=60)
    DATA = R.json()

    import html
    to_ret = html.unescape(DATA['expandtemplates']['wikitext'])
    return to_ret


def gross_clean(text, regexes, convert_through_api, convert_to_text_dictionary: DictProxy, request_session1,

                v_lock):
    text = re.sub(r'&nbsp;', ' ', text)

    # html comments
    text = re.sub(r'(<!--.*?-->)', "", text, flags=re.DOTALL)

    # refs
    text = re.sub(r'<ref( name ?= ?\"?(.*?)\"?)?((>(.*?)<\/ref>)|(\ ?\/>))', r'', text, flags=re.DOTALL)

    # files
    text = recursive_clean(text, {"[["}, {"]]"}, {"[[File:", "[[Image:"})
    text = text.strip()

    ####### BEGIN the idea here is to remove blocks {{...}} from the beginning of the document
    prev_len_text = len(text)
    text = recursive_clean_klim(text, '{{', '}}', '{{', '}}', min_nr_levels=1, look_pre_in_beginning=True)
    text = text.strip()
    while len(text) != prev_len_text:
        prev_len_text = len(text)
        text = recursive_clean_klim(text, '{{', '}}', '{{', '}}', min_nr_levels=1, look_pre_in_beginning=True)
        text = text.strip()
    ###### END the idea here is to remove blocks {{...}} from the end of the document

    text = recursive_clean(text, {"{{"}, {"}}"}, {"{{Infobox"})
    text = recursive_clean(text, {"{{"}, {"}}"}, {"{{Taxobox"})

    text = recursive_clean(text, {"{|"}, {"|}"})

    convert_all_regex = regexes['compiled_all_finder']
    last_span_added_pos = 0
    to_ret = ''

    for curr_found_convert in convert_all_regex.finditer(text):
        span_start = curr_found_convert.start()
        span_end = curr_found_convert.end()
        convert_text = curr_found_convert.group()
        expanded_convert = ''
        if convert_through_api:
            if convert_text not in convert_to_text_dictionary:
                v_lock.acquire()
                done = False
                nr_retries = 0
                if convert_text not in convert_to_text_dictionary:
                    while not done and nr_retries < 5:
                        nr_retries += 1
                        try:
                            expanded_convert = expand_convert_api(convert_text, request_session1)
                            convert_to_text_dictionary[convert_text] = expanded_convert
                            done = True
                        except ConnectionError as e:  # This is the correct syntax
                            logger.error('kzaporoj ConnectionError happened: ')
                            logger.error(traceback.format_exc())
                            logger.error('kzaporoj resetting the request_session1')
                            logger.error('nr_retries is as follows: %s' % nr_retries)
                            expanded_convert = convert_text
                            sleep(5)
                        except:
                            logger.error('nr_retries is as follows: ', nr_retries)
                            logger.error('some other except for connection')
                            logger.error(traceback.format_exc())
                            sleep(5)
                v_lock.release()
            else:
                expanded_convert = convert_to_text_dictionary[convert_text]
        else:
            pass  # just leaves expanded_convert in '', deleting it from the text
        to_ret += text[last_span_added_pos:span_start]
        to_ret += ' ' + expanded_convert
        last_span_added_pos = span_end

    if last_span_added_pos > 0:
        to_ret += text[last_span_added_pos:]
        text = to_ret

    text = re.sub(r'(?i)\[\[wikt\:(.*?)\|.*?\]\]', lambda m: m.group(1), text)

    # BEGIN IAST cases: todo: remove (?i) ?? --> which ignores upper/lower case
    text = re.sub(r'(?i)\{\{IAST\|(.*?)\}\}', lambda m: m.group(1), text)
    # END IAST cases

    # BEGIN IPA
    text = re.sub(r'(?i)\{\{IPA\|(.*?)\}\}', lambda m: m.group(1), text)
    # END IAST cases

    # BEGIN transl
    text = re.sub(r'(?i)\{\{transl\|[a-z\-]{2,10}?\|[A-Za-z]+?\|(.*?)\}\}', lambda m: m.group(1), text)
    text = re.sub(r'(?i)\{\{transl\|[a-z\-]{2,10}?\|(.*?)\}\}', lambda m: m.group(1), text)
    # END transl

    # BEGIN cases like "CO2" in "and [[greenhouse gas|{{CO2}} emissions]]"
    text = re.sub(r'(?i)\{\{([A-Za-z0-9]{1,15})?\}\}', lambda m: m.group(1), text)
    # END transl

    # BEGIN for color
    text = re.sub(r'(?i)\{\{color\|(.*?)\|(.*?)\}\}', lambda m: m.group(2), text)
    # END for color

    # begin the nowrap
    text = re.sub(r'(?i)\{\{nowrap\|(.*?)\}\}', lambda m: m.group(1), text)
    text = re.sub(r'(?i)\{\{nobr\|(.*?)\}\}', lambda m: m.group(1), text)
    text = re.sub(r'(?i)\{\{nobreak\|(.*?)\}\}', lambda m: m.group(1), text)
    # end the nowrap template

    # begin for mvar
    text = re.sub(r'(?i)\{\{mvar\|(.*?)\}\}', lambda m: m.group(1), text)
    # end for mvar

    # begin the chem
    # {{chem|C|''n''|H|2''n''&nbsp;+&nbsp;2}}
    text = re.sub(r'(?i)\{\{chem\|(.*?)\}\}',
                  lambda m: m.group(1).replace('\'', '').replace('|', ''), text)
    # end the chem template

    # BEGIN Unicode cases: todo: remove (?i) ?? --> which ignores upper/lower case
    text = re.sub(r'(?i)\{\{Unicode\|(.*?)\}\}', lambda m: m.group(1), text)

    # END Unicode cases

    # removes the sub/sup-indices mark
    text = re.sub(r'(?i)<sub>(.*?)</sub>', lambda m: m.group(1), text)
    text = re.sub(r'(?i)<sup>(.*?)</sup>', lambda m: m.group(1), text)

    # ignore all the text after and including references; not relevant for disambiguation;
    if '==References==' in text:
        index_of_references = text.index('==References==')
        text = text[:index_of_references]
    elif '== References ==' in text:
        index_of_references = text.index('== References ==')
        text = text[:index_of_references]

    return text


def clean_text_from_link_markers(input_text):
    parsed: Wikicode = mwparserfromhell.parse(input_text)
    text = parsed.strip_code()

    text = re.sub(r'\([\s\,\.]*\)', ' ', text)

    text = re.sub(r' +', ' ', text)

    return text.strip()


def fine_clean(text):
    text = re.sub(r'<br\s?/?>', '\n', text)

    text = re.sub(r'\{\{cite(.*?)\}\}', r'', text, flags=re.DOTALL + re.IGNORECASE)

    text = re.sub(r'<ref(.*?)(name ?= ?\"?(.*?)\"?)?((>(.*?)<\/ref>)|(\ ?\/>))', r'', text, flags=re.DOTALL)

    text = re.sub(r'\([\s\,\.]*\)', ' ', text)

    text = re.sub(r' +', ' ', text)

    return text.strip()


if __name__ == "__main__":
    # cleaned = simple_clean('is [[Karman Line|{{convert|100|km|mi|abbr=on}}]] a weapon?')
    infobox = "before infobox {{Infobox settlement\n|name                   = Booterstown\n|" \
              "other_name             = {{pad top italic|Baile an Bhóthair}}\n|" \
              "settlement_type        = Suburb of Dublin\n|image_skyline          = \n|" \
              "image_caption          = \n|pushpin_map            = Ireland\n|" \
              "pushpin_label_position = right\n|pushpin_map_caption    = Location in Ireland\n|" \
              "coordinates_display    = inline,title\n|coordinates_region     = IE\n|" \
              "subdivision_type       = Country\n|subdivision_name       =  Ireland\n|" \
              "subdivision_type1      =  Province\n|subdivision_name1      =  Leinster\n|" \
              "subdivision_type3      =  County\n|subdivision_name3      =  Dun Laoghaire-Rathdown\n|" \
              "established_title      = \n|established_date       = \n|" \
              "leader_title1          =  Dáil Éireann\n|leader_name1           =  Dún Laoghaire\n|" \
              "unit_pref              = Metric\n|area_footnotes         = \n|area_total_km2         = \n|" \
              "population_as_of       = 2006\n|population_footnotes   = \n|population_total       = \n|" \
              "population_urban       = 2975\n|population_density_km2 = auto\n|" \
              "timezone1              =  WET \n|utc_offset1            = +0\n|" \
              "timezone1_DST          =  IST ( WEST) \n|utc_offset1_DST        = -1\n|" \
              "latd                   = 53.3087\n|longd                  = -6.1964\n|" \
              "coordinates_format     = dms\n|coordinates_type       = dim:100000_region:IE\n|" \
              "elevation_footnotes    = \n|elevation_m            = \n|area_code              = 01, +353 1\n|" \
              "blank_name             =  Irish Grid Reference\n|blank_info             = {{iem4ibx|O201304}}\n|" \
              "website                = \n|footnotes              = \n}} after infobox"

    cleaned_infobox = recursive_clean(infobox, {"{{"}, {"}}"}, {"{{Infobox"})
    logger.info('cleaned infobox: %s' % cleaned_infobox)
