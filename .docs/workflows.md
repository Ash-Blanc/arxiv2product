# User Workflows

`arxiv2product` offers multiple ways to interact with the pipeline, depending on your needs.

## 1. CLI: Analyze a Specific Paper
Use this when you have an arXiv ID or URL.
```bash
cd cli
uv run arxiv2product 2603.09229
```
**Output**: `products_2603_09229.md`

## 2. CLI: Topic Discovery (PASA-style)
Use this when you don't have a paper yet but have a research topic.
```bash
# Enable in .env
ENABLE_PAPER_SEARCH=1

uv run arxiv2product "self-adapting language models"
```
**Process**: Finds 10 relevant papers, scores them, picks the best one, and runs the full pipeline.

## 3. CLI: Competitor Intelligence
A post-pipeline deep dive into specific ideas. This is separated because web automation (crawling competitor landing pages) is slower and more expensive.
```bash
uv run arxiv2product-compete products_2603_09229.md --ideas 1,2
```
**Tools used**: Parallel.ai (search) and Tinyfish (browser automation).

## 4. Local API Service
Run the pipeline as a background service.
```bash
cd cli
uv run arxiv2product-api
```
**Key Endpoints**:
- `POST /reports`: Queue a new paper for analysis.
- `GET /reports/{jobId}`: Poll for status and results.
- `GET /dashboard/{userId}`: View history.

## 5. Testing & Verification
We use standard Python `unittest`.
```bash
cd cli
uv run python -m unittest discover -s tests
```
*Note: Most tests mock network/model calls to ensure reliable CI/CD.*
