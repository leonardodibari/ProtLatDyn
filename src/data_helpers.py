# src/data_helpers.py

def format_name(first_name, last_name):
    """
    A reusable function to properly format a full name.
    """
    if not first_name or not last_name:
        return "Name Error"
        
    formatted = f"{first_name.strip().capitalize()} {last_name.strip().capitalize()}"
    return formatted

def calculate_age_in_months(years):
    return years * 12