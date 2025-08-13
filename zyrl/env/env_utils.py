def sorted_file_by_trade_time(trade_time_list):
    sorted_trade_time = []
    night_trade_time = None
    for trade_time in sorted(trade_time_list):
        if "am" in trade_time:
            if night_trade_time is not None:
                sorted_trade_time.append(night_trade_time)
                night_trade_time = None
            sorted_trade_time.append(trade_time)

        elif "night" in trade_time:
            if night_trade_time is not None:
                sorted_trade_time.append(night_trade_time)
            night_trade_time = trade_time
        else:
            sorted_trade_time.append(trade_time)
            if night_trade_time is not None:
                sorted_trade_time.append(night_trade_time)
                night_trade_time = None
    return sorted_trade_time
