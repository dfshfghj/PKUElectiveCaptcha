package cn.pku.elective.captcha;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.ArrayList;
import java.util.List;

public final class Main {
    private Main() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) usage();
        String command = args[0];
        Map<String, String> options = options(args);
        switch (command) {
            case "generate" -> generate(options);
            case "verify" -> verify(Path.of(args.length > 1 ? args[1] : required(options, "manifest")));
            default -> usage();
        }
    }

    private static void generate(Map<String, String> options) throws Exception {
        GeneratorConfig defaults = GeneratorConfig.defaults();
        Map<String, String> file = new HashMap<>();
        if (options.containsKey("config")) file.putAll(YamlConfig.load(Path.of(options.get("config"))));
        else {
            try { file.putAll(YamlConfig.loadResource("application.yml")); }
            catch (Exception ignored) { /* packaged builds may intentionally omit defaults */ }
        }
        int width = integer(options, file, "width", defaults.width());
        int height = integer(options, file, "height", defaults.height());
        String charset = value(options, file, "charset", defaults.charset());
        String format = value(options, file, "format", defaults.format());
        long seed = Long.parseLong(options.getOrDefault("seed", Long.toString(defaults.seed())));
        Path output = Path.of(required(options, "output"));
        int count = integer(options, file, "count", 1);
        String lengthOption = options.containsKey("length")
                ? options.get("length") : value(options, file, "lengths", "4,5");
        String profileOption = value(options, file, "profiles", "clean,line,shadow,line-shadow");
        String prefix = options.getOrDefault("prefix", "");
        List<String> focusPairs = options.containsKey("focus-pairs")
                ? List.of(options.get("focus-pairs").split(",")) : List.of();
        List<GeneratorConfig> configs = new ArrayList<>();
        for (String lengthText : lengthOption.split(",")) {
            int length = Integer.parseInt(lengthText.trim());
            for (String profileText : profileOption.split(",")) {
                configs.add(new GeneratorConfig(width, height, length, charset, format,
                        profileText.trim(), seed));
            }
        }
        if (options.containsKey("flatten")) {
            KaptchaGenerator.Result result = new KaptchaGenerator().generateFlattened(configs, output, count,
                    options.containsKey("force"), focusPairs, prefix);
            System.out.printf("generated=%d per_combination=%d elapsed_ms=%d flattened=true output=%s%n",
                    result.count(), count, result.elapsedMillis(), output);
            return;
        }
        int total = 0;
        long elapsed = 0;
        for (GeneratorConfig config : configs) {
            Path combination = output.resolve("length-" + config.length()).resolve("profile-" + config.profile());
            KaptchaGenerator.Result result = new KaptchaGenerator().generate(config, combination, count,
                    options.containsKey("force"));
            total += result.count();
            elapsed += result.elapsedMillis();
        }
        System.out.printf("generated=%d per_combination=%d elapsed_ms=%d combinations=%s x %s output=%s%n",
                total, count, elapsed, lengthOption, profileOption, output);
    }

    private static void verify(Path manifest) throws Exception {
        if (!Files.isRegularFile(manifest)) throw new IllegalArgumentException("manifest not found: " + manifest);
        var lines = Files.readAllLines(manifest);
        if (lines.size() < 2) throw new IllegalArgumentException("manifest has no samples");
        Path root = manifest.getParent();
        int checked = 0;
        Set<String> filenames = new HashSet<>();
        for (String line : lines.subList(1, lines.size())) {
            String[] fields = line.split(",", -1);
            if (fields.length < 15) throw new IllegalArgumentException("invalid manifest row: " + line);
            Path image = root.resolve(fields[1]);
            if (!filenames.add(fields[1])) throw new IllegalArgumentException("duplicate filename: " + fields[1]);
            if (!Files.isRegularFile(image)) throw new IllegalArgumentException("missing image: " + image);
            if (!fields[13].matches("[0-9a-z]+")) throw new IllegalArgumentException("invalid label: " + fields[13]);
            BufferedImage decoded = ImageIO.read(image.toFile());
            if (decoded == null || decoded.getWidth() != Integer.parseInt(fields[10])
                    || decoded.getHeight() != Integer.parseInt(fields[11])) {
                throw new IllegalArgumentException("image dimensions do not match manifest: " + image);
            }
            if (!KaptchaGenerator.sha256(image).equals(fields[14])) throw new IllegalArgumentException("hash mismatch: " + image);
            checked++;
        }
        System.out.printf("verified=%d manifest=%s%n", checked, manifest);
    }

    private static Map<String, String> options(String[] args) {
        Map<String, String> result = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            if (!args[i].startsWith("--")) continue;
            String key = args[i].substring(2);
            result.put(key, i + 1 < args.length && !args[i + 1].startsWith("--") ? args[++i] : "true");
        }
        return result;
    }

    private static int integer(Map<String, String> options, Map<String, String> file, String key, int fallback) {
        return Integer.parseInt(value(options, file, key, Integer.toString(fallback)));
    }

    private static String value(Map<String, String> options, Map<String, String> file, String key, String fallback) {
        return options.getOrDefault(key, file.getOrDefault("generator." + key, fallback));
    }

    private static String required(Map<String, String> options, String key) {
        if (!options.containsKey(key)) throw new IllegalArgumentException("missing --" + key);
        return options.get(key);
    }

    private static void usage() {
        throw new IllegalArgumentException("usage: generate --output DIR --count N [--seed N --charset STRING --lengths 4,5 --profiles clean,line,shadow,line-shadow --flatten --format png|jpeg] | verify MANIFEST");
    }
}
