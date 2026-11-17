# College Buddy

College Buddy is a local-first retrieval-augmented admission guidance app. It
searches uploaded college documents and answers with source filenames so users
can verify the guidance.

## Run locally

1. Use Python 3.10 or later.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

4. Open the local URL printed by Streamlit and upload UTF-8 `.txt` or `.md`
   admission documents in the sidebar. The combined upload limit is 5 MB.

Without uploaded files, the app uses general demonstration guidance, not
college-specific facts. Uploaded content is held in the app session and is not
saved permanently. Verify deadlines and requirements with the college's
official admissions office.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `project.py`, and `requirements.txt` to a GitHub repository.
   Include `README.md` if you want the setup instructions alongside the code.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and select **Create app**.
3. Choose the repository and branch, set the app file path to `app.py`, and
   select **Deploy**.
4. After deployment, open the app URL and upload the college guidance files
   through the sidebar.

The uploaded files are session data; redeploying or starting another session
does not make them permanent. The app does not require an API key or send
questions to an external language-model service.
