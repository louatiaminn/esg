import argparse, random, os
import pandas as pd
import numpy as np

COMPANIES = ["Acme Corp", "Greenfield Ltd", "Suncrest Inc", "NovaSolutions", "Atlas Manufacturing"]
ENV_TERMS = ["carbon emissions", "renewable energy", "solar", "wind", "waste management", "recycling", "net-zero"]
SOC_TERMS = ["employee safety", "diversity", "inclusion", "labor rights", "community engagement"]
GOV_TERMS = ["board", "audit", "compliance", "transparency", "executive compensation"]

VAGUE_TEMPLATES = [
    "{c} is committed to sustainability and being a responsible company.",
    "We prioritize being green and supporting sustainable practices at {c}.",
    "{c} continues to foster a culture of responsibility."
]

ENV_TEMPLATES = [
    "{c} reduced its {term} by {num}% after investing in {tech}.",
    "{c} set a target to source {num}% renewable energy by {year}."
]

SOC_TEMPLATES = [
    "{c} launched a {program} to improve {term} across all sites.",
    "{c} reported a {num}% increase in workforce {term} year-over-year."
]

GOV_TEMPLATES = [
    "{c} updated its executive compensation policy after shareholder review.",
    "The board at {c} approved new {policy} to strengthen {term}."
]

NONESG_TEMPLATES = [
    "{c} launched a new product that features {feature}.",
    "{c} reported quarterly revenue growth of {num}% and exceeded guidance."
]

def rand_company(): return random.choice(COMPANIES)
def rand_num(low=5, high=70): return random.randint(low, high)
def rand_year(): return random.randint(2025, 2035)
def pick_term(terms): return random.choice(terms)

def fill_template(template, company):
    return template.format(
        c=company,
        term=pick_term(ENV_TERMS + SOC_TERMS + GOV_TERMS),
        tech=random.choice(["solar panels", "wind turbines", "energy-efficient systems"]),
        program=random.choice(["diversity initiative", "safety training program"]),
        policy=random.choice(["audit policy", "ethics policy", "transparency policy"]),
        feature=random.choice(["longer battery life", "improved UX"]),
        num=rand_num(),
        year=rand_year()
    )

def generate_one(combo):
    E,S,G = combo
    c = rand_company()
    parts = []
    if E:
        parts.append(fill_template(random.choice(ENV_TEMPLATES + VAGUE_TEMPLATES), c))
    if S:
        parts.append(fill_template(random.choice(SOC_TEMPLATES), c))
    if G:
        parts.append(fill_template(random.choice(GOV_TEMPLATES), c))
    if not (E or S or G):
        parts.append(fill_template(random.choice(NONESG_TEMPLATES), c))
    if random.random() < 0.2:
        parts.append("The initiative is part of the company's broader strategic plan.")
    text = " ".join(parts)
    return {"text": text, "E": int(E), "S": int(S), "G": int(G), "nonESG": int(not (E or S or G))}

def generate_dataset(n=2000, seed=42):
    random.seed(seed); np.random.seed(seed)
    # default distribution
    combos = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1),(0,0,0)]
    probs  = [0.20,0.12,0.12,0.06,0.06,0.04,0.02,0.38]
    rows = []
    for _ in range(n):
        combo = random.choices(combos, weights=probs, k=1)[0]
        rows.append(generate_one(combo))
    return pd.DataFrame(rows)

def save_splits(df, out_dir, val_frac=0.1, test_frac=0.1, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n*test_frac); n_val = int(n*val_frac)
    df.iloc[:n-n_val-n_test].to_csv(os.path.join(out_dir, "synthetic_train.csv"), index=False)
    df.iloc[n-n_val-n_test:n-n_test].to_csv(os.path.join(out_dir, "synthetic_val.csv"), index=False)
    df.iloc[n-n_test:].to_csv(os.path.join(out_dir, "synthetic_test.csv"), index=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    args = p.parse_args()
    df = generate_dataset(n=args.n, seed=args.seed)
    save_splits(df, args.out_dir, args.val_frac, args.test_frac, args.seed)
    print("Saved synthetic dataset to", args.out_dir)
