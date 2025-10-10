# connect_to_jira.py

import os
from jira import JIRA

# --- Configuration: FILL IN YOUR DETAILS HERE ---

# 1. Your Jira site URL
JIRA_SERVER = 'https://chaoticbunch69.atlassian.net' 

# 2. The email you used to sign up for Jira
JIRA_EMAIL = 'chaoticbunch69@gmail.com' # <--- PUT YOUR EMAIL HERE

# 3. The API Token you created and saved
JIRA_API_TOKEN = 'ATATT3xFfGF0d4AXYKVXBuSbCD3PMWWUqcbbNba09ESCu0GD0D6SQ9Dfk6Z3T1m3qkzsS6aRoq6g3LGmAvmg2NvZL_cxuV9seb7X-PeuKWtumeSdQ6fZCV2ARyjJPaoE_dPcyztd9qZRIUZKlQL28XGsqPrF3RNUAPe__BjCrMT3CPTuE6Agbrs=EB440A98' # <--- PUT YOUR API TOKEN HERE

# --- End of Configuration ---


# Connect to Jira
print(f"Connecting to Jira server at {JIRA_SERVER}...")
try:
    jira_client = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    print("✅ Successfully connected to Jira!")
except Exception as e:
    print(f"❌ Failed to connect to Jira: {e}")
    exit()


# Search for Issues
# 4. Your Project Key from the URL
project_key = "GT" 
jql_query = f'project = {project_key}'

print(f"Searching for issues with JQL: '{jql_query}'")
issues_found = jira_client.search_issues(jql_query)

if not issues_found:
    print("No issues found. Make sure you have created an issue in your project.")
else:
    print(f"Found {len(issues_found)} issue(s).")
    # Process Issues and Download Attachments
    for issue in issues_found:
        print(f"\nProcessing Issue: {issue.key} - {issue.fields.summary}")
        
        if issue.fields.attachment:
            for attachment in issue.fields.attachment:
                filename = attachment.filename
                print(f"  - Found attachment: {filename}")
                
                # Create a directory to store downloads
                if not os.path.exists('jira_attachments'):
                    os.makedirs('jira_attachments')
                
                download_path = os.path.join('jira_attachments', filename)

                # Download the file
                with open(download_path, 'wb') as f:
                    f.write(attachment.get())
                print(f"    - ✅ Downloaded to '{download_path}'")
        else:
            print("  - No attachments found for this issue.")