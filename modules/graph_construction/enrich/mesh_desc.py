import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
api_key = os.getenv("NCBI_API_KEY")  

def parse_mesh_html(mesh_id: str, html: str) -> dict:
    """
    Parse NCBI MeSH HTML and extract name + scope note.
    """

    soup = BeautifulSoup(html, "html.parser")
    result = {"mesh_id": mesh_id, "name": None, "scope_note": None}

    content = soup.find("div", class_="rprt") or soup.find("div", id="maincontent")
    if content is None:
        return result

    h1 = content.find("h1")
    if h1:
        result["name"] = h1.get_text(strip=True).replace("[Supplementary Concept]", "").strip()
        sibling = h1.find_next_sibling()
        while sibling:
            text = sibling.get_text(strip=True)
            if any(kw in text for kw in ["Year introduced", "PubMed search", "Tree Number", "Subheading"]):
                break
            if text:
                result["scope_note"] = text
                break
            sibling = sibling.find_next_sibling()

    return result

async def fetch_one(session: aiohttp.ClientSession, mesh_id: str, semaphore: asyncio.Semaphore, api_key: str = api_key) -> dict:
    url = f"https://www.ncbi.nlm.nih.gov/mesh/?term={mesh_id}"
    if api_key:
        url += f"&api_key={api_key}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; mesh-fetcher/1.0)"}

    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                html = await resp.text()
                return parse_mesh_html(mesh_id, html)
        except Exception as e:
            print(f"Error fetching {mesh_id}: {e}")
            return {"mesh_id": mesh_id, "name": None, "scope_note": None}

async def fetch_mesh_batch(mesh_ids: list, max_concurrent: int = 10, api_key: str = api_key) -> list:
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, mid, semaphore, api_key=api_key) for mid in mesh_ids]
        return await asyncio.gather(*tasks)

def fetch_mesh_many(mesh_ids: list, max_concurrent: int = 5) -> list:
    """Sync wrapper — call this from a notebook or script."""
    return asyncio.run(fetch_mesh_batch(mesh_ids, max_concurrent))

# Start Running
    nest_asyncio.apply()

    CHECKPOINT_FILE = 'nodes_checkpoint.csv'
    CHECKPOINT_IDX_FILE = 'checkpoint_idx.txt'
    BATCH_SIZE = 50 

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_IDX_FILE) and os.path.exists(CHECKPOINT_FILE):
        nodes = pd.read_csv(CHECKPOINT_FILE)
        with open(CHECKPOINT_IDX_FILE, 'r') as f:
            start_idx = int(f.read().strip()) + 1
        print(f"Resuming from index {start_idx}")
    else:
        nodes['description'] = None
        start_idx = 0

    pending = nodes.iloc[start_idx:]

    # Process in batches so we checkpoint regularly
    for batch_start in tqdm(range(0, len(pending), BATCH_SIZE), desc='Batches'):
        batch = pending.iloc[batch_start : batch_start + BATCH_SIZE]
        mesh_ids = batch['mesh_id'].tolist()

        # Fetch concurrently
        results = await fetch_mesh_batch(mesh_ids, max_concurrent=10)

        for result in results:
            nci_text = result.get('scope_note')
            if (nci_text is not None
                    and not nci_text.startswith("MeSH Unique ID")
                    and not ("structure" in nci_text.lower() or "first source" in nci_text.lower())):
                nodes.loc[nodes['mesh_id'] == result['mesh_id'], 'description'] = nci_text[0].upper() + nci_text[1:]

        # Checkpoint after each batch
        last_idx = start_idx + batch_start + len(batch) - 1
        nodes.to_csv(CHECKPOINT_FILE, index=False)
        with open(CHECKPOINT_IDX_FILE, 'w') as f:
            f.write(str(last_idx))

    print("Done!")

# Add to the Graph after we finish with the csv
    BATCH_SIZE = 500

    query = """
        UNWIND $rows AS row

        MERGE (node: Test {id: row.id})
        SET node.description = row.description
        """

    # Process in batches

    for i in tqdm(range(0, len(nodes), BATCH_SIZE), desc="Batch processing"):

        batch = nodes.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id"],
                "description": row["description"]
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(i + BATCH_SIZE))
