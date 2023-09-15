from datetime import datetime
import inflect

def number_to_words(n):
    # Convert a number into words.
    # There are many ways to do this, and one common approach is to use the `inflect` library.
    # For brevity, I won't implement the entire logic here, but will mention the use of `inflect`.
    p = inflect.engine()
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

