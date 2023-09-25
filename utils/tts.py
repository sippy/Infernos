import re
from datetime import datetime
import inflect

def number_to_words(n):
    # Convert a number into words.
    # There are many ways to do this, and one common approach is to use the `inflect` library.
    # For brevity, I won't implement the entire logic here, but will mention the use of `inflect`.
    p = inflect.engine()
    if isinstance(n, re.Match):
        n = n.group(0)
    return p.number_to_words(n)

def get_ordinal(n):
    # Convert a number into its ordinal representation.
    p = inflect.engine()
    return p.ordinal(n)

def human_readable_time():
    now = datetime.now()

    # Days and months are straightforward
    day_name = now.strftime('%A')
    month_name = now.strftime('%B')

    # Convert day of the month and year to words
    day_of_month = number_to_words(int(now.strftime('%d')))
    year = number_to_words(int(now.strftime('%Y')))
    year = year.replace('-', ' ')

    # Convert hour and minute to words
    if now.hour < 12:
        time_period = "morning"
    elif 12 <= now.hour < 17:
        time_period = "afternoon"
    elif 17 <= now.hour < 20:
        time_period = "evening"
    else:
        time_period = "night"

    hour = number_to_words(now.hour % 12 or 12)
    if now.minute != 0:
        minute = number_to_words(now.minute)
        current_time = f"{hour} {minute}"
    else:
        current_time = f"{hour} o'clock"

    return f"Today is {day_name} {get_ordinal(day_of_month)} of {month_name} {year}, {current_time} in the {time_period}."

import requests

wq_fixes = (
    ('<.*?>', ''), ('\[.*?\]', ''),
    (r'\s+', ' '), ('Mr[.]', 'Mister'),
    ('Dr[.]', 'Doctor'), ('Drs.', 'Doctors'), ('["]', ''),
    (r'\d+', number_to_words), ('H.A.L.', '"H" "A" "L"'),
    ('Thomas A[.] Anderson','Thomas A Anderson'),
    ('i-sy,', 'iiisy,'), ('i-zy,', 'iiizzy,'),
    ('Agent Smith As', 'As'), ('.*edit[]] ', ''),
    ('Trinity: .*', ''), ('ar-riage', 'arrrrrrriage'),
    ('Dialogue The ', 'The '), ('cra-zy', 'craaaazy',),
    ('[%] ', ' percent '),
)

class ECFail(Exception):
    pass

def extract_content(url, start_pattern, end_pattern):
    headers = {
        'User-Agent': 'Wget/1.20.3 (linux-gnu)'
    }
    response = requests.get(url, headers=headers)
    print(url, response)
    if response.status_code != 200 or len(response.text) == 0:
        raise ECFail(f"Failed to retrieve URL. Status code: {response.status_code}")

    content = response.text
    s=content.find(start_pattern)

    i = 0
    pattern = re.compile(rf"{start_pattern}(.*?){end_pattern}", re.DOTALL)
    matches = pattern.findall(content)
    clean = [(re.compile(p), r) for p, r in wq_fixes]

    matches = [m.split(':', 1)[1] for m in matches]
    for cl, rv in clean:
        matches = [re.sub(cl, rv, m).strip() for m in matches]
    return matches

def wq_getscript(film, character, section=1):
    BASE_URL = "https://en.wikiquote.org/w/index.php"
    film = film.replace(' ', '_')
    fsuf = '_(film)'
    url = f"{BASE_URL}?title={film}&section={section}"
    start_pattern = rf">{character}<"
    end_pattern = r'</dd>'
    try:
        cont = extract_content(url, start_pattern, end_pattern)
        if len(cont) == 0:
            raise ECFail("nope")
    except ECFail as ex:
        if not film.endswith(fsuf):
            film += fsuf
            url = f"{BASE_URL}?title={film}&section={section}"
            cont = extract_content(url, start_pattern, end_pattern)
        else:
            raise
    return cont

def hal_set():
    contents = wq_getscript('2001: A Space Odyssey', 'HAL')
    return [s.replace('. ', '.|') for s in contents]

def bender_set(season=1):
    contents = wq_getscript(f'Futurama/Season_{season}', 'Bender')
    return [s for s in contents if len(s) > 16]

def smith_set():
    contents = wq_getscript('The Matrix', 'Agent Smith', section=4)
    hp = 'As you can see, we'
    hack = contents[0].split(hp)
    if len(hack) <= 2:
        raise Exception("cleanme, hack is not needed perhaps anymore")
    contents[0] = hp + hack[-1]
    return [s.replace('. ', '.|') for s in contents]

def t900_set():
    contents = wq_getscript('The Terminator', 'Terminator', section=3)
    return [s.replace('. ', '.|') for s in contents]
