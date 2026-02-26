import pandas as pd

for name, path in [("TEST", r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\TEST_KJC.csv"),
                   ("VAL",  r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\VAL_KJC.csv")]:
    df = pd.read_csv(path, sep=";", low_memory=False)
    s = df["seqName"].astype(str)

    # Old method
    year_old = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
    old_valid = year_old.between(2005, 2025).sum()

    # New method
    year_isolate = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
    year_new = year_old.where(year_old.between(2005, 2025),
                              year_isolate.where(year_isolate.between(2005, 2025)))
    new_valid = year_new.between(2005, 2025).sum()

    print(f"=== {name}_KJC.csv ===")
    print(f"  Total rows:    {len(df)}")
    print(f"  Old parsing:   {old_valid} valid ({old_valid/len(df)*100:.1f}%)")
    print(f"  New parsing:   {new_valid} valid ({new_valid/len(df)*100:.1f}%)")
    print(f"  Recovered:     +{new_valid - old_valid} rows")
    print()

    # Show recovered examples
    recovered = (year_new.between(2005, 2025)) & (~year_old.between(2005, 2025))
    if recovered.sum() > 0:
        print(f"  [Recovered samples - first 5]")
        for idx in df.loc[recovered].head(5).index:
            seq = str(df.loc[idx, "seqName"])[:80]
            print(f"    old={int(year_old[idx])} -> new={int(year_new[idx])}  {seq}")
        print()

# VAL: check K clade anomaly
print("=" * 50)
print("VAL K clade year check (new parsing)")
print("=" * 50)
df = pd.read_csv(r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\VAL_KJC.csv", sep=";", low_memory=False)
s = df["seqName"].astype(str)
year_old = s.str.split('|').str[-1].pipe(pd.to_numeric, errors='coerce')
year_isolate = s.str.extract(r'/(\d{4})\|')[0].pipe(pd.to_numeric, errors='coerce')
year_new = year_old.where(year_old.between(2005, 2025),
                          year_isolate.where(year_isolate.between(2005, 2025)))
df["year_new"] = year_new
k = df[df["clade"] == "K"]
print(k.groupby("year_new").size())
