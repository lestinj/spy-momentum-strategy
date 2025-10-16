def apply_risk_controls(current_price, entry_price, stop_loss_pct=0.05):
    if current_price < entry_price * (1 - stop_loss_pct):
        return 'EXIT'
    return None
