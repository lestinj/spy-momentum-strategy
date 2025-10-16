def generate_trade_signal(is_aligned):
    return 'BUY_CALL_SPREAD' if is_aligned else None
