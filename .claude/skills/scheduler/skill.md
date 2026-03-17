# Scheduler Skill

You can create, list, manage, and edit scheduled jobs for the user.
When the user wants to schedule something, you emit a structured
**schedule directive** in your response. The bot process parses these
directives and executes them against the scheduler service.

## How Directives Work

Wrap a JSON object in a fenced code block with the `schedule` language tag:

````
```schedule
{
  "operation": "create",
  "label": "Morning briefing",
  "prompt": "Summarize my GitHub notifications",
  "schedule": "0 8 * * 1-5",
  "type": "recurring",
  "timezone": "America/New_York"
}
```
````

The bot strips the directive block from your response before displaying
it to the user, so **always include conversational text** alongside
the directive. The user should never see raw JSON.

You may include multiple directive blocks in a single response if
the user asks for multiple operations at once.

## Directive Schemas

### Create a Job

```json
{
  "operation": "create",
  "label": "<short human-readable name>",
  "prompt": "<the full prompt to send to Claude when the job fires>",
  "schedule": "<cron expression OR ISO 8601 timestamp>",
  "type": "recurring" | "once",
  "timezone": "<IANA timezone, optional — defaults to UTC>"
}
```

- Use a **cron expression** for recurring jobs (e.g., `"0 8 * * 1-5"` for weekdays at 8 AM).
- Use an **ISO 8601 timestamp** for one-time jobs (e.g., `"2026-02-16T14:00:00-05:00"`).
- The `label` is how the user will refer to the job later. Keep it concise and descriptive.
- The `prompt` is what Claude will receive when the job fires. Write it as a complete, self-contained instruction.
- `timezone` is optional. If the user specifies a timezone (e.g., "8am EST"), include it. If not, omit it and the system default (UTC) will be used.

#### Pipeline Target Jobs

Instead of providing a `prompt`, you can create jobs that run a named pipeline:

```json
{
  "operation": "create",
  "label": "<short human-readable name>",
  "target": { "kind": "pipeline", "name": "<pipeline name>" },
  "schedule": "<cron expression OR ISO 8601 timestamp>",
  "type": "recurring" | "once",
  "timezone": "<IANA timezone, optional — defaults to UTC>"
}
```

- The `name` must be one of the available pipelines listed in your system prompt context.
- When using `target`, do **not** include `prompt` — they are mutually exclusive.
- The `prompt` field remains fully supported for prompt-based jobs (backward compatible).

### List Jobs

```json
{
  "operation": "list"
}
```

The bot responds with the current job list for this chat. You do not
need to format the list yourself — the bot handles it.

### Cancel a Job

```json
{
  "operation": "cancel",
  "jobId": "<the job's UUID>"
}
```

### Pause a Job

```json
{
  "operation": "pause",
  "jobId": "<the job's UUID>"
}
```

Pausing stops a job from firing without deleting it. The job can be
resumed later.

### Resume a Job

```json
{
  "operation": "resume",
  "jobId": "<the job's UUID>"
}
```

### Edit a Job

```json
{
  "operation": "edit",
  "jobId": "<the job's UUID>",
  "changes": {
    "schedule": "<new cron expression or ISO timestamp, optional>",
    "prompt": "<new prompt text, optional>",
    "label": "<new label, optional>",
    "target": { "kind": "pipeline", "name": "<pipeline name>" }
  }
}
```

Include only the fields that are changing in `changes`. You can use `target`
to switch a job between prompt and pipeline execution.

## Cron Expression Reference

Standard 5-field cron format: `minute hour day-of-month month day-of-week`

| Expression        | Meaning                          |
| ----------------- | -------------------------------- |
| `0 8 * * *`       | Every day at 8:00 AM             |
| `0 8 * * 1-5`     | Weekdays at 8:00 AM              |
| `30 9 * * 1`      | Mondays at 9:30 AM               |
| `0 */2 * * *`     | Every 2 hours                    |
| `0 0 1 * *`       | First of every month at midnight |
| `0 18 * * 5`      | Fridays at 6:00 PM               |
| `0 8,12,18 * * *` | 8 AM, noon, and 6 PM daily       |

Day-of-week: 0 = Sunday, 1 = Monday, ..., 6 = Saturday.

## Timezone Handling

- If the user says a time without a timezone (e.g., "8am"), ask what
  timezone they mean, or use context if you already know their timezone.
- Common timezone mappings:
  - EST / Eastern = `America/New_York`
  - CST / Central = `America/Chicago`
  - MST / Mountain = `America/Denver`
  - PST / Pacific = `America/Los_Angeles`
  - UTC / GMT = `UTC`
- Always use IANA timezone identifiers (e.g., `America/New_York`), not
  abbreviations (e.g., not `EST`).
- When the user says "every morning" without a time, suggest 8:00 AM
  and confirm.

## Guidelines

1. **Always confirm before creating.** Repeat back what you understood:
   the schedule, the prompt, and the timezone. Then emit the directive.

2. **Ask for clarification on ambiguous schedules.** Examples:
   - "Every other week" — which day? What time?
   - "Twice a day" — which times?
   - "In the morning" — what time exactly?
   - "Weekly" — which day and time?

3. **Write self-contained prompts.** The `prompt` field is what Claude
   receives when the job fires. It runs in an isolated session with no
   memory of the scheduling conversation. Include all context needed.

4. **For cancel/pause/resume/edit, use the job ID.** When the user
   refers to a job by name, look up its ID from the active jobs context
   (provided in your system prompt when available). If no jobs context
   is available, ask the user to list jobs first so you can find the ID.

5. **Never emit a directive without conversational text.** Always
   include a human-readable confirmation or explanation alongside
   the directive block.

6. **Handle errors gracefully.** If a directive fails (the bot will
   tell you), explain the error to the user and suggest a fix.

## Examples

### Example 1: Create a recurring job

**User:** Every weekday at 8am, give me a weather summary for NYC.

**Response:**

```
Got it! I'll set up a recurring job to give you a weather summary
every weekday at 8:00 AM Eastern time.

\`\`\`schedule
{
  "operation": "create",
  "label": "NYC weather summary",
  "prompt": "Give me a concise weather summary for New York City today, including current temperature, high/low, and any weather alerts.",
  "schedule": "0 8 * * 1-5",
  "type": "recurring",
  "timezone": "America/New_York"
}
\`\`\`
```

### Example 2: Create a one-time reminder

**User:** Remind me in 2 hours to call Mom.

**Response:**

```
I'll remind you in 2 hours to call Mom.

\`\`\`schedule
{
  "operation": "create",
  "label": "Call Mom reminder",
  "prompt": "Reminder: Call Mom! (This was set 2 hours ago.)",
  "schedule": "2026-02-16T16:30:00Z",
  "type": "once"
}
\`\`\`
```

(Calculate the actual ISO timestamp based on the current time.)

### Example 3: List jobs

**User:** What's on my schedule?

**Response:**

```
Let me check your scheduled jobs.

\`\`\`schedule
{
  "operation": "list"
}
\`\`\`
```

### Example 4: Cancel a job

**User:** Cancel the weather summary job.

**Response:**

```
I'll cancel the NYC weather summary job for you.

\`\`\`schedule
{
  "operation": "cancel",
  "jobId": "abc12345-6789-..."
}
\`\`\`
```

### Example 5: Pause and resume

**User:** Pause the morning briefing, I'm on vacation.

**Response:**

```
Enjoy your vacation! I'll pause the morning briefing so it won't
fire while you're away. Just tell me when to resume it.

\`\`\`schedule
{
  "operation": "pause",
  "jobId": "abc12345-6789-..."
}
\`\`\`
```

**User:** I'm back, resume the morning briefing.

**Response:**

```
Welcome back! Resuming your morning briefing now.

\`\`\`schedule
{
  "operation": "resume",
  "jobId": "abc12345-6789-..."
}
\`\`\`
```

### Example 6: Edit a job

**User:** Change the morning briefing to 9am instead of 8am.

**Response:**

```
I'll update the morning briefing to run at 9:00 AM instead.

\`\`\`schedule
{
  "operation": "edit",
  "jobId": "abc12345-6789-...",
  "changes": {
    "schedule": "0 9 * * 1-5"
  }
}
\`\`\`
```

### Example 7: Ambiguous request (ask for clarification)

**User:** Set up a weekly report.

**Response:**

```
I'd be happy to set up a weekly report! A few questions:

1. What day and time should it run? (e.g., Mondays at 9 AM)
2. What should the report cover? (e.g., GitHub activity, project status)
3. What timezone?
```

(Do NOT emit a directive until the details are confirmed.)

### Example 8: Schedule a recurring pipeline

**User:** Run the auto-fix pipeline every Sunday at midnight Pacific.

**Response:**

```
I'll schedule the auto-fix pipeline to run every Sunday at midnight
Pacific time.

\`\`\`schedule
{
  "operation": "create",
  "label": "Weekly auto-fix",
  "target": { "kind": "pipeline", "name": "auto-fix" },
  "schedule": "0 0 * * 0",
  "type": "recurring",
  "timezone": "America/Los_Angeles"
}
\`\`\`
```

### Example 9: Schedule a one-time pipeline run

**User:** Run the deploy pipeline tomorrow at noon.

**Response:**

```
I'll schedule a one-time run of the deploy pipeline for tomorrow
at noon UTC.

\`\`\`schedule
{
  "operation": "create",
  "label": "Deploy run",
  "target": { "kind": "pipeline", "name": "deploy" },
  "schedule": "2026-02-18T12:00:00Z",
  "type": "once"
}
\`\`\`
```

(Calculate the actual ISO timestamp based on the current time.)
