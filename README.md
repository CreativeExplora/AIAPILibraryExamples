# Financial Modeling Assistant

A Gemini-powered CLI tool for creating comprehensive financial transaction nodes from business plans and strategies.

## Features

- Upload PDF documents containing business plans or financial strategies
- Automatically generate structured financial nodes with transactions, constraints, and timing
- Support for complex business scenarios with 50-100+ nodes
- Structured JSON output matching database schema
- Thought signature logging for transparency

## Setup

1. Install dependencies:
```bash
pip install google-genai pydantic python-dotenv
```

2. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python3 gemini.py
```

## Usage

### Upload Files
```
upload:/path/to/business_plan.pdf
```

### Create Nodes
```
create:make nodes based on the uploaded plan
```

The system will generate comprehensive financial nodes including:
- Node names and descriptions
- Transaction details (debits/credits)
- Constraints (categories, departments)
- Timing rules (start/end dates, recurrence)
- Expected values

### Normal Chat
Ask questions or have conversations about the uploaded documents without the `create:` prefix.

## Data Structure

### Node Schema
Each node contains:
- `node_name`: Name of the financial node
- `constraints`: List of constraint strings
- `transaction`: List of transactions with debits and credits
- `transaction_description`: Human-readable description
- `absolute_start_utc`: Start timestamp (ISO 8601)
- `absolute_end_utc`: End timestamp (optional)
- `start_offset_rule`: Start time offset rule (optional)
- `end_offset_rule`: End time offset rule (optional)
- `recurrence_rule`: Recurrence pattern (optional)
- `expected_value`: Numeric expected value

### Transaction Structure
```json
{
  "name": "Frontend engineer",
  "debits": [
    {"amount": "100", "account": "Salaries Expense"}
  ],
  "credits": [
    {"amount": "100", "account": "Cash"}
  ]
}
```

## Commands

- `upload:/path/to/file` - Upload a local PDF file
- `create:your request` - Generate nodes based on input
- `quit` - Exit the application

## Example Session

```
You: upload:business_strategy.pdf
Uploaded: business_strategy.pdf

You: create:generate nodes for all budget items and headcount

[GENERATING NODES]...

[CREATED] 47 nodes:

--- Node 1 ---
node_name: Q1_Marketing_Spend
constraints: ['marketing', 'Q1']
transaction: [{'name': 'Marketing Campaign', 'debits': [{'amount': '50000', 'account': 'Marketing Expense'}], 'credits': [{'amount': '50000', 'account': 'Cash'}]}]
...

[COMPLETE] 47 nodes created.
```

## Model

Uses `gemini-2.5-flash` with structured JSON output for reliable, schema-compliant node generation.
