package cn.pku.elective.captcha;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

final class YamlConfig {
    private YamlConfig() {}

    static Map<String, String> load(Path path) throws IOException {
        try (InputStream input = Files.newInputStream(path)) {
            return parse(new String(input.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    static Map<String, String> loadResource(String name) throws IOException {
        try (InputStream input = YamlConfig.class.getClassLoader().getResourceAsStream(name)) {
            if (input == null) throw new IOException("config resource not found: " + name);
            return parse(new String(input.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    private static Map<String, String> parse(String content) {
        Map<String, String> values = new HashMap<>();
        Deque<String> parents = new ArrayDeque<>();
        int previousIndent = 0;
        for (String raw : content.split("\\R")) {
            if (raw.isBlank() || raw.trim().startsWith("#")) continue;
            int indent = raw.indexOf(raw.trim());
            String line = raw.trim();
            int colon = line.indexOf(':');
            if (colon < 0) throw new IllegalArgumentException("invalid YAML line: " + raw);
            while (indent < previousIndent && !parents.isEmpty()) parents.pop();
            String key = line.substring(0, colon).trim();
            String value = line.substring(colon + 1).trim();
            if (value.isEmpty()) {
                parents.push(key);
            } else {
                StringBuilder full = new StringBuilder(key);
                for (String parent : parents) full.insert(0, parent + ".");
                values.put(full.toString(), unquote(value));
            }
            previousIndent = indent;
        }
        return values;
    }

    private static String unquote(String value) {
        if (value.length() >= 2 && ((value.startsWith("\"") && value.endsWith("\""))
                || (value.startsWith("'") && value.endsWith("'")))) {
            return value.substring(1, value.length() - 1);
        }
        return value;
    }
}
