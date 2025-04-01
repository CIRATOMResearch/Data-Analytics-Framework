current_df = None


def get_current_df():
    global current_df
    return current_df


def set_current_df(df):
    global current_df
    current_df = df