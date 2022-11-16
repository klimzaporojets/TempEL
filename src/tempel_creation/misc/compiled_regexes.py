import re

compiled_regexes = {
    'compiled_mention_finder': re.compile(r'\[\[+.*?\]\]+[A-Za-z]*'),
    'compiled_country_in_link': re.compile(r'^[\:]{0,1}([a-z\-]{2,15})\:'),
    'compiled_convert_finder': re.compile(r'(?i)\{\{Convert\|(.*?)\}\}'),
    'compiled_subatomic_particle_finder': re.compile(r'(?i)\{\{SubatomicParticle\|(.*?)\}\}'),
    'compiled_all_finder': re.compile(r'(?i)((\{\{SubatomicParticle\|(.*?)\}\})|(\{\{Convert\|(.*?)\}\}))')
}
