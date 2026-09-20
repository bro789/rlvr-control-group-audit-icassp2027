# arXiv 提交清单

> **状态：暂缓。** cs.LG 需要 endorsement，走完要几天，而 ICASSP 9/23 截稿。
> 论文改为只指向本仓库，不依赖 arXiv。若日后拿到背书，本清单和
> `arxiv_submission.tar.gz` 仍然可用，直接照着走即可。

上传文件：`arxiv_submission.tar.gz`（340 KB，36 个文件，解包后净室编译 38 页通过）

---

## 0. 先确认一件可能卡住你的事

**endorsement（背书）。** 如果这是你在 `cs.LG` 的第一次提交，arXiv 可能要求一位已在该分类发表过的人背书，走完流程要几天。ICASSP 是 9/23 截稿，所以：

- 现在就登录 <https://arxiv.org/user> 看一眼有没有提示需要背书
- 需要的话，找导师或同门（在 cs.LG 发过的）走 endorsement
- **短版脚注现在只指 GitHub，不依赖 arXiv**，所以就算 arXiv 卡住也不影响投稿

## 1. 开始

<https://arxiv.org/submit> → Start New Submission

## 2. 分类

| 字段 | 填 |
|---|---|
| Primary | **cs.LG**（Machine Learning） |
| Cross-list | **cs.CL**（Computation and Language） |

## 3. 许可

**CC BY 4.0** —— 与 GitHub 仓库的 `LICENSE` 一致。选它是因为论文的立场就是"数据和脚本可复用"，挑更严的许可会自相矛盾。

## 4. 上传

选 tarball 上传，arXiv 会自己跑 pdflatex。

- **不要**另外上传 PDF —— arXiv 优先用源码编译，传了 PDF 反而可能冲突
- 包里已含 `main.bbl`（arXiv **不跑 bibtex**，没它参考文献全空）
- 包里**没有**任何 `.aux`/`.log`/`.pdf`，这是 arXiv 要求的

上传后它会自动编译，点 "View PDF" 核对一遍再继续。

## 5. Metadata

**Title**

```
Something Simpler Wins, But Not Always the Same One: Auditing RLVR Control Groups Across Three Backbone Scales
```

**Authors**（arXiv 格式，逗号分隔）

```
Keyi Li, Yihao He, Quanyi Li, Guanghui Liu, Xiaoyu Lu, Linying Jiang
```

**Abstract** —— 用 `ARXIV_ABSTRACT.txt`（1725 字符，在 1920 上限内；PDF 里那版三段的超限了，所以这里是压缩版，内容口径一致）

**Comments**

```
Extended version of a paper submitted to ICASSP 2027. 38 pages. Includes
complete per-group statistics, a control-group aggregator sensitivity analysis,
a cross-fitted audit of the decision rule for selection bias, an
inference-engine equivalence gate, and a follow-up that bounds the 7B result by
compute budget. Code, data, pre-registration and deviation logs:
https://github.com/bro789/rlvr-control-group-audit-icassp2027
```

**ACM Class**（可选）`I.2.6; I.2.7`

## 6. 提交之后

- **编号在提交时就分配**（形如 2609.xxxxx），不必等公告
- 公告通常是下一个工作日；提交可能被 moderator 暂扣，那就要等
- 拿到编号后告诉我，我加回短版脚注 —— `main_4p.tex` 里留了注释说明加在哪

## 7. 两处不要忘

1. 挂完之后，把 arXiv 编号补进 GitHub 仓库 `README.md` 末尾那行 `<!-- Fill in once the arXiv identifier is assigned. -->`
2. 短版脚注加回 arXiv 编号后**要重编一次**，并确认仍是 5 页、第 5 页只有参考文献（现在版面是满的，加任何东西都可能溢出）
