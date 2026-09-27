def parse_ground_truth(df):
    out={}
    for _,r in df.iterrows():
        ids=[x.strip() for x in str(r["matched_entity_ids"]).split(",") if x.strip()]
        out[str(r["source1_entity_id"])]=ids
    return out
