def yes_or_no(question):
    reply = str(input(f'{question} (y/n):').lower().strip())
    if reply == 'y':
        return True
    elif reply == 'n':
        return False
    else:
        print('Invalid input. Not proceeding.')
        return False