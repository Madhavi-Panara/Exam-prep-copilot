# Build plan (deadline: Nov 15)

## MVP
- [x] Upload PDF/TXT/MD, extract text, chunk, embed, store in ChromaDB (BM25 fallback)
- [x] Ask a question, get a cited answer from your notes
- [x] Generate a quiz on a topic and grade the answers

## Bonus
- [x] Weak-topics tracker
- [x] Past-paper topic importance analysis
- [x] Study plan generator

## Remaining (your side)
- [ ] Add your API key and run the app end to end with your real notes
- [ ] Try it on 2-3 real subjects; note any bad questions or answers and tune `backend/app/prompts.py`
- [ ] Push to GitHub, record the demo video, fill in the submission form using `docs/SUBMISSION.md`

## Schedule
- Nov 11: run end to end, fix anything that breaks
- Nov 12-13: test with real notes, tune prompts, polish UI
- Nov 14: demo video, README check, submission write-up
- Nov 15: buffer, submit early
