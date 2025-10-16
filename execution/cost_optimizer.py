def choose_optimal_strike(options_df, budget, target_delta=0.5):
    options_df['delta_diff'] = (options_df['delta'] - target_delta).abs()
    sorted_df = options_df.sort_values('delta_diff')
    for _, row in sorted_df.iterrows():
        if row['ask'] <= budget:
            return row
    return None