Workflow

User question (English / Hinglish)
        │
        ▼
1. Agent's AI model (GPT-5-Mini) + Perspective (the agent's rules)
   - Is it about IDCUBE Helpdesk?  No → polite refusal
   - What is asked: one ticket / list of tickets / management numbers?
   - Fills in defaults (period, all statuses…) instead of asking
        │
        ▼
2. Calls 1–2 of the 3 Odoo tools, always sending every input
   ├─ odoo_ticket_lookup  → one ticket: details, SLA, conversation,
   │                        history, health check & risk flags
   ├─ odoo_ticket_list    → filtered ticket list + exact total
   └─ odoo_helpdesk_report→ counts, SLA %, resolution time, breakdowns
                            (engineer/team/region/category/customer/month),
                            comparison with the previous period,
                            key findings, aged backlog
        │
        ▼
3. Inside each tool (Python)
   - Converts the period to IST dates, only tickets from 1 Jan 2026
   - Matches names even if partial or misspelt (customer/SI/end customer,
     engineer, team, category)
   - Reads live data from Odoo (read-only: search_read / read / count)
   - Calculates the numbers and returns them
        

than 

4. Excel (only for quarter/year reports or when asked)
   Spreadsheet tools: Create Workbook → Write Data for each sheet
   → file shows in the chat for download
        │
        ▼
5. Answer: headline → table → insights → risks → recommendations
   → "data as of" time. Humble tone, English or Hinglish.
I have also attached/included the workflow for your better understanding. Could you please review it and share your feedback?
