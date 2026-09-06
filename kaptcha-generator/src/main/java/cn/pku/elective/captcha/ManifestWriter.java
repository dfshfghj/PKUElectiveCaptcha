package cn.pku.elective.captcha;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;

final class ManifestWriter implements AutoCloseable {
    private final BufferedWriter writer;

    ManifestWriter(Path path, GeneratorConfig config, boolean force) throws IOException {
        this(path, force);
    }

    ManifestWriter(Path path, boolean force) throws IOException {
        writer = Files.newBufferedWriter(path, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        writer.write("timestamp,filename,split,profile,seed,sample_index,sample_seed,generator_version,kaptcha_version,java_version,width,height,format,label,sha256\n");
    }

    void write(GeneratorConfig config, int index, long sampleSeed, String filename,
               String label, Path image) throws IOException {
        writer.write(String.join(",", Instant.now().toString(), "images/" + filename, "unsplit", config.profile(),
                Long.toString(config.seed()), Integer.toString(index), Long.toString(sampleSeed),
                "0.1.0", "2.3.2", System.getProperty("java.version"),
                Integer.toString(config.width()), Integer.toString(config.height()), config.format(), label,
                KaptchaGenerator.sha256(image)));
        writer.write('\n');
    }

    @Override public void close() throws IOException { writer.close(); }
}
