# Git 备忘录（没有 AI 也能自己上传）

> 打开终端的方法：文件资源管理器进入本项目文件夹 → **地址栏输入 `powershell` 回车**（终端直接就在当前目录），或右键文件夹 → 「在终端中打开」。

## 核心概念（就 4 个）

| 概念 | 人话 |
|---|---|
| 仓库 | 项目文件夹里多了个隐藏的 `.git` 文件夹（**别删它**，删了历史就没了） |
| commit | 存档。每次存档都有编号和说明，随时能时光倒流 |
| push | 把本地存档上传到 GitHub 云端 |
| pull | 把云端的新存档下载回来（换电脑/多人协作用） |

## 三板斧（日常改完代码就走一遍）

```powershell
cd D:\College_learning\pde-visualization    # 进入项目（用地址栏打开终端则跳过这行）
git status         # 先看状态：红色 = 还没暂存的新改动
git add .          # 把改动放进暂存区（选好要存档的内容）
git commit -m "说明：这次改了什么"            # 存档
git push           # 上传到 GitHub，完事
```

**每行都可以敲 `git status` 确认**——它是只读的，随便敲不会出事。

## 新项目从零上传（三步 + 网页建仓）

```powershell
# ① 在项目文件夹里打开终端，初始化并首次存档
git init
git add .
git commit -m "初始提交"
git branch -M main

# ② 去网页：github.com → 右上角 + → New repository
#    名字和文件夹一样，Public，⚠️不勾 README/.gitignore/license → Create

# ③ 回到终端，连接并上传（URL 换成你的）
git remote add origin https://github.com/lovely-lxt/项目名.git
git push -u origin main
```

## 常用查询

| 命令 | 作用 |
|---|---|
| `git status` | 现在什么状态（最常用） |
| `git log --oneline` | 看存档列表 |
| `git diff` | 看具体改了什么（add 之前用） |
| `git pull` | 把云端存档拉下来 |

## 坑与常识

- **push 弹浏览器授权是正常的**：第一次会让你登录 GitHub 并点 Authorize，之后就记住了。注意：在本人的正常终端里不用任何额外设置（AI 的自动化终端才需要特殊开关）。
- 报错就读**最后几行英文**，基本都是能看懂的（"repository not found" = 网页仓库没建或名字不一致）。
- commit message 写清楚，推荐「动词开头」：`新增：xxx` / `修改：xxx` / `修复：xxx`。
- 永远不要在项目文件夹里再 `git init` 一次。
