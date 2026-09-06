package cn.pku.elective.captcha;

import com.google.code.kaptcha.impl.DefaultKaptcha;
import com.google.code.kaptcha.util.Config;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.SplittableRandom;
import java.util.List;

public final class KaptchaGenerator {
    public record Result(int count, long elapsedMillis) {}

    public Result generate(GeneratorConfig rawConfig, Path output, int count, boolean force)
            throws IOException {
        GeneratorConfig config = rawConfig.validate();
        if (count < 1) throw new IllegalArgumentException("count must be positive");
        Path manifest = output.resolve("manifest.csv");
        if (Files.exists(manifest) && !force) {
            throw new IllegalArgumentException("manifest already exists; use --force to replace the output");
        }
        Files.createDirectories(output.resolve("images"));
        long start = System.nanoTime();
        try (ManifestWriter writer = new ManifestWriter(manifest, config, force)) {
            SplittableRandom random = new SplittableRandom(config.seed());
            for (int index = 0; index < count; index++) {
                long sampleSeed = random.nextLong();
                String label = label(config.charset(), config.length(), new SplittableRandom(sampleSeed));
                DefaultKaptcha producer = new DefaultKaptcha();
                producer.setConfig(new Config(config.toKaptchaProperties(label)));
                BufferedImage image = producer.createImage(label);
                String filename = "%08d_%s.%s".formatted(index, label, config.format().toLowerCase());
                Path imagePath = output.resolve("images").resolve(filename);
                try (OutputStream stream = Files.newOutputStream(imagePath)) {
                    if (!ImageIO.write(image, config.format(), stream)) {
                        throw new IOException("no ImageIO writer for " + config.format());
                    }
                }
                writer.write(config, index, sampleSeed, filename, label, imagePath);
            }
        }
        return new Result(count, (System.nanoTime() - start) / 1_000_000);
    }

    public Result generateFlattened(List<GeneratorConfig> configs, Path output, int count, boolean force)
            throws IOException {
        return generateFlattened(configs, output, count, force, List.of(), "");
    }

    public Result generateFlattened(List<GeneratorConfig> configs, Path output, int count, boolean force,
                                    List<String> focusPairs, String filenamePrefix) throws IOException {
        if (configs.isEmpty()) throw new IllegalArgumentException("at least one configuration is required");
        if (count < 1) throw new IllegalArgumentException("count must be positive");
        Path manifest = output.resolve("manifest.csv");
        if (Files.exists(manifest) && !force) {
            throw new IllegalArgumentException("manifest already exists; use --force to replace the output");
        }
        Path images = output.resolve("images");
        Files.createDirectories(images);
        long start = System.nanoTime();
        int total = 0;
        try (ManifestWriter writer = new ManifestWriter(manifest, force)) {
            for (GeneratorConfig rawConfig : configs) {
                GeneratorConfig config = rawConfig.validate();
                SplittableRandom random = new SplittableRandom(config.seed());
                for (int index = 0; index < count; index++) {
                    long sampleSeed = random.nextLong();
                    String label = focusPairs.isEmpty()
                            ? label(config.charset(), config.length(), new SplittableRandom(sampleSeed))
                            : focusedLabel(config.charset(), config.length(), focusPairs.get(index % focusPairs.size()),
                                    new SplittableRandom(sampleSeed));
                    DefaultKaptcha producer = new DefaultKaptcha();
                    producer.setConfig(new Config(config.toKaptchaProperties(label)));
                    BufferedImage image = producer.createImage(label);
                    String filename = "%slength-%d_%s_%08d_%s.%s".formatted(
                            filenamePrefix.isBlank() ? "" : filenamePrefix + "_", config.length(), config.profile(),
                            index, label, config.format().toLowerCase());
                    Path imagePath = images.resolve(filename);
                    try (OutputStream stream = Files.newOutputStream(imagePath)) {
                        if (!ImageIO.write(image, config.format(), stream)) {
                            throw new IOException("no ImageIO writer for " + config.format());
                        }
                    }
                    writer.write(config, index, sampleSeed, filename, label, imagePath);
                    total++;
                }
            }
        }
        return new Result(total, (System.nanoTime() - start) / 1_000_000);
    }

    static String label(String charset, int length, SplittableRandom random) {
        StringBuilder result = new StringBuilder(length);
        for (int i = 0; i < length; i++) result.append(charset.charAt(random.nextInt(charset.length())));
        return result.toString();
    }

    static String focusedLabel(String charset, int length, String pair, SplittableRandom random) {
        if (pair.length() != 3 || pair.charAt(1) != ':') {
            throw new IllegalArgumentException("focus pair must look like m:n: " + pair);
        }
        char first = pair.charAt(0), second = pair.charAt(2);
        if (length < 2 || charset.indexOf(first) < 0 || charset.indexOf(second) < 0) {
            throw new IllegalArgumentException("focus pair is incompatible with charset/length: " + pair);
        }
        char[] result = new char[length];
        for (int i = 0; i < length; i++) result[i] = charset.charAt(random.nextInt(charset.length()));
        int firstPosition = random.nextInt(length);
        int secondPosition = random.nextInt(length - 1);
        if (secondPosition >= firstPosition) secondPosition++;
        result[firstPosition] = first;
        result[secondPosition] = second;
        return new String(result);
    }

    static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(Files.readAllBytes(path));
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new AssertionError(exception);
        }
    }
}
