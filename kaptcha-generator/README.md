# Kaptcha Generator

使用 Kaptcha 生成离线 OCR 合成数据

```bash
mvn test
mvn -q package
java -jar target/kaptcha-generator-0.1.0.jar generate \
  --count 1500 --output ../data/synthetic/kaptcha-v1 \
  --seed 42 --lengths 4,5 \
  --profiles white-shadow-line,blue-shadow-line,white-outline-line,white-outline \
  --flatten --format png
java -jar target/kaptcha-generator-0.1.0.jar verify \
  ../data/synthetic/kaptcha-v1/manifest.csv
```

如需将所有长度和干扰组合放在同一个目录，增加 `--flatten`：

```bash
java -jar target/kaptcha-generator-0.1.0.jar generate \
  --count 1000 --output ../data/synthetic/kaptcha-flat \
  --lengths 4,5 --profiles clean,line,shadow,line-shadow \
  --flatten --format png
```

扁平化输出位于 `images/`，文件名形如 `length-4_line_00000000_483fy.png`

默认配置位于 `src/main/resources/application.yml`

默认使用实际字符集 `2345678abcdefgmnpwxy`
