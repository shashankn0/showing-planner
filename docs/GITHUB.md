# Putting this project on GitHub (from zero to pushed)

Everything here was checked on 2026-10-08. The command sequence was rehearsed end to end against
a local stand-in for GitHub (first push, edit/commit/push, fresh clone, re-pointing a remote, the
author-email rewrite). The parts that happen on github.com itself follow GitHub's own docs.

## Where you are right now

- The project is already a Git repository on branch `main` with a full history and **no remote**,
  so nothing is on GitHub yet.
- It contains **no** listing data, photos, keys, databases or model weights, now or anywhere in its
  history. `.gitignore` blocks `data/`, `.env`, `*.db`, and model weights.
- 94 files, about 1 MB.

## Decide three things first

1. **Repository name.** Use `showing-planner` (matches the folder). GitHub allows only letters,
   digits, `.`, `-` and `_`, up to 100 characters. Pick something you're happy to have in the URL:
   `https://github.com/YOUR-USERNAME/showing-planner`.
2. **Public or private.** Private is safest while you're experimenting. Streamlit Community Cloud's
   free hosting is simplest with a **public** repo (private repos may need extra authorization,
   which I couldn't confirm). Either way, no listing data is in the repo.
3. **Your commit email.** Every commit carries an author name and email, and GitHub shows them
   publicly. Right now that is `shashankn0 <snellutla26@gmail.com>`. If you don't want your real
   email public, do **Step 1** below before the first push (it's only clean before the first push).

Also note: no `LICENSE` file exists. Without one, others can see the code on a public repo but
have no permission to reuse it. Add one only if you want that (MIT is a common permissive choice);
that's your call.

## Step 0: check Git

```powershell
git --version
git config --global user.name
git config --global user.email
```

If either `user.` line is blank: `git config --global user.name "Your Name"` and the same for
`user.email`.

## Step 1 (optional but recommended): hide your email

1. On github.com: profile picture, **Settings**, **Emails**. Tick **Keep my email addresses
   private**. GitHub then shows your private "noreply" address on that page; it looks like
   `12345678+yourusername@users.noreply.github.com`.
2. Use it for this project, and re-author the existing commits (safe only because nothing has been
   pushed yet; this rewrites history, so never do it after others have cloned):

   ```powershell
   cd C:\Users\snell\showing-planner
   git config user.name "Your Name"
   git config user.email "12345678+yourusername@users.noreply.github.com"
   git rebase -r --root --exec "git commit --amend --no-edit --reset-author"
   git log --format="%an <%ae>" | sort -u
   ```

   The last line should now show only the noreply address. (Rehearsed: all 15 commits re-authored,
   messages and the code itself unchanged.) Commit messages from this session also carry a
   `Co-Authored-By: Claude` line, which GitHub will display; that's accurate attribution.

## Step 2: create the empty repository on github.com

1. Click the **+** at the top right, then **New repository**.
2. **Owner:** you. **Repository name:** `showing-planner`. **Description:** optional.
3. Choose **Public** or **Private**.
4. **Leave every extra unchecked:** no README, no `.gitignore`, no license. Your project already
   has them, and adding them on GitHub creates a conflicting history that makes the first push
   fail.
5. Click **Create repository**. Keep the page open; it shows your repository's URL.

## Step 3: connect and push

```powershell
cd C:\Users\snell\showing-planner
git status                       # should say "nothing to commit, working tree clean"
git remote add origin https://github.com/YOUR-USERNAME/showing-planner.git
git branch -M main               # already main; harmless
git push -u origin main
```

`-u` links your local `main` to GitHub's, so from now on plain `git push` and `git pull` work.
Refresh the GitHub page: your files and 15+ commits should be there. Check locally:

```powershell
git status -sb                   # first line should be:  ## main...origin/main
```

### If it asks you to sign in

- **A GitHub login is already stored on this PC** (your Git is set to keep logins in a plain-text
  file in your home folder), so the push may succeed with no prompt. That stored login must have
  permission to write to the new repo; a token limited to other repositories won't.
- To sign in fresh: a browser window opens (Git Credential Manager) or Git asks for a username and
  password. The "password" must be a **token**, not your GitHub password:
  1. github.com, profile picture, **Settings**, **Developer settings**, **Personal access tokens**,
     **Fine-grained tokens**, **Generate new token**.
  2. Name it, set an expiry, **Repository access: Only select repositories**, choose
     `showing-planner`.
  3. **Permissions, Repository permissions, Contents: Read and write.** Generate, copy it once.
  4. Username: your GitHub username. Password: paste the token.
- **Safer storage:** Git Credential Manager is already installed here (v2.7.0). Switch to it so
  tokens live in Windows' encrypted credential store instead of a plain-text file:

  ```powershell
  git config --global credential.helper manager
  ```

  You may also want to rotate (revoke and recreate) whatever token is in the old plain-text file at
  `C:\Users\snell\.git-credentials`, then delete that file.

## The everyday loop (commit and push changes)

```powershell
cd C:\Users\snell\showing-planner
git status                       # what changed? make sure no data/, .env or photos are listed
git add -A                       # stage everything that changed (ignored files are skipped)
git diff --staged --stat         # last look at exactly what will be saved
git commit -m "Short sentence saying what and why"
git push
```

Good messages are one clear sentence: "Add school filter", "Fix price parsing for K/M". Small,
frequent commits make mistakes easy to undo.

Useful, safe commands:

```powershell
git log --oneline -10            # recent history
git diff                         # unstaged changes
git restore path\to\file         # throw away unsaved edits to one file (can't be undone)
git restore --staged path\to\file    # unstage a file
git revert <commit-id>           # undo an old commit with a new commit (safe after pushing)
git pull                         # get changes made elsewhere
```

Avoid `git push --force` and `git reset --hard` unless you understand them; they can erase work.

### Working on a feature without touching `main` (optional)

```powershell
git switch -c feature/school-filter
# ...edit, add, commit...
git push -u origin feature/school-filter
```

Then on GitHub click **Compare & pull request**, review, and merge. Then `git switch main` and
`git pull`.

## What must never be committed

`.gitignore` already blocks these; `git status` is your safety net.

| Never commit | Why |
|---|---|
| `data/` (listings, photos, database, embeddings, trained models) | MLS-derived and licensed; confirm with your broker |
| `.env`, API keys, tokens | Anyone can read a public repo, and bots scan for keys within minutes |
| Test photos from Wikimedia or anywhere else | Not yours to redistribute |

Check any file: `git check-ignore -v data\app.db .env` (it prints the rule that ignores it).

**If you commit a secret by accident:** first **revoke or rotate the key** at its provider (assume
it's already stolen), then remove it from history with a tool like `git filter-repo` or BFG.
Deleting the file in a new commit does **not** remove it from history.

## On another computer (or after a disk failure)

```powershell
git clone https://github.com/YOUR-USERNAME/showing-planner.git
cd showing-planner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

Your local `data/` (listings, photos, trained models) does **not** travel through GitHub; copy that
folder yourself if you need it.

## When something goes wrong

| Message | What it means and the fix |
|---|---|
| `remote origin already exists` | A remote is already set. Re-point it: `git remote set-url origin https://github.com/YOUR-USERNAME/showing-planner.git` |
| `rejected ... fetch first` or `non-fast-forward` on the first push | You created the GitHub repo with a README/license. Either delete the repo and recreate it empty, or `git pull --rebase origin main` then `git push` |
| `Authentication failed` or `403` | The token is wrong, expired, or lacks **Contents: Read and write** on this repo. Make a new fine-grained token (above) |
| `src refspec main does not match any` | No commits yet or the branch isn't `main`: `git branch -M main` and make sure `git log` shows commits |
| `LF will be replaced by CRLF` warnings | Harmless Windows line-ending notices; `.gitattributes` already handles it |
| `file exceeds 100 MB` | A big file got staged. `git restore --staged <file>`, add it to `.gitignore` |
| `fatal: not a git repository` | You're in the wrong folder: `cd C:\Users\snell\showing-planner` |

## After it's pushed

To publish the demo website for free, follow [DEPLOY.md](DEPLOY.md) (Option A), which starts from a
GitHub repo like the one you just made.
