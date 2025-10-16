# Isaac_Lab

very_good~

## 上传到 GitHub 的步骤

由于此环境无法直接访问外部网络，你需要在本地终端执行以下命令将代码推送到 GitHub：

```bash
git remote add origin <你的仓库地址>  # 如果尚未配置远程仓库
git fetch origin
# 确认当前分支（例如 work）已经包含你需要的提交后

git push origin work
```

如果远程仓库使用其他分支名称，请把上面的 `work` 替换成目标分支的名称。
