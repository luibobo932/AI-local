import os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "numpy", "gradio"])
shutil.rmtree("repo", ignore_errors=True)  # luôn tải bản mới nhất
subprocess.run(["git", "clone", "--depth", "1", "-b",
                "claude/local-language-model-3g7ije",
                "https://github.com/luibobo932/AI-local", "repo"])
exec(open("repo/hf_space/_chat_app.py").read())
