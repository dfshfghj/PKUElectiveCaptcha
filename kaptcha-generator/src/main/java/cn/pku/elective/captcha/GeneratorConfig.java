package cn.pku.elective.captcha;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;

public record GeneratorConfig(int width, int height, int length, String charset,
                              String format, String profile, long seed) {
    public static GeneratorConfig defaults() {
        return new GeneratorConfig(130, 52, 5,
                "2345678abcdefgmnpwxy", "png", "line-shadow", 20260906L);
    }

    public GeneratorConfig validate() {
        if (width < 1 || height < 1 || length < 1 || charset == null || charset.isEmpty()) {
            throw new IllegalArgumentException("width, height, length and charset must be positive");
        }
        if (charset.chars().distinct().count() != charset.length()) {
            throw new IllegalArgumentException("charset must not contain duplicate characters");
        }
        if (!format.equalsIgnoreCase("png") && !format.equalsIgnoreCase("jpeg")) {
            throw new IllegalArgumentException("format must be png or jpeg");
        }
        if (!profile.equals("clean") && !profile.equals("line") && !profile.equals("shadow")
                && !profile.equals("line-shadow") && !profile.equals("default")
                && !profile.equals("white-shadow-line") && !profile.equals("blue-shadow-line")
                && !profile.equals("white-outline-line") && !profile.equals("white-outline")) {
            throw new IllegalArgumentException("unsupported profile: " + profile);
        }
        return this;
    }

    public Properties toKaptchaProperties(String label) {
        Properties properties = new Properties();
        properties.setProperty("kaptcha.image.width", Integer.toString(width));
        properties.setProperty("kaptcha.image.height", Integer.toString(height));
        properties.setProperty("kaptcha.textproducer.char.length", Integer.toString(label.length()));
        properties.setProperty("kaptcha.textproducer.char.string", charset);
        properties.setProperty("kaptcha.border", "no");
        boolean line = profile.equals("line") || profile.equals("line-shadow") || profile.equals("default")
                || profile.equals("white-shadow-line") || profile.equals("blue-shadow-line")
                || profile.equals("white-outline-line");
        properties.setProperty("kaptcha.noise.impl", line
                ? "com.google.code.kaptcha.impl.DefaultNoise"
                : "com.google.code.kaptcha.impl.NoNoise");
        String obscurificator = switch (profile) {
            case "line", "white-outline-line", "white-outline", "blue-shadow-line", "white-shadow-line" ->
                    profile.equals("line") ? "cn.pku.elective.captcha.LineGimpy" :
                            (profile.equals("white-outline-line") || profile.equals("white-outline")
                                    ? "cn.pku.elective.captcha.WeakOutlineGimpy"
                                    : "com.google.code.kaptcha.impl.ShadowGimpy");
            case "shadow", "line-shadow" -> "com.google.code.kaptcha.impl.ShadowGimpy";
            default -> "com.google.code.kaptcha.impl.WaterRipple";
        };
        properties.setProperty("kaptcha.obscurificator.impl", obscurificator);
        if (profile.startsWith("white-")) {
            properties.setProperty("kaptcha.textproducer.font.color", "255,255,255");
            properties.setProperty("kaptcha.background.clear.from", "170,175,185");
            properties.setProperty("kaptcha.background.clear.to", "205,210,220");
        } else if (profile.startsWith("blue-")) {
            properties.setProperty("kaptcha.textproducer.font.color", "40,100,215");
            properties.setProperty("kaptcha.background.clear.from", "238,243,251");
            properties.setProperty("kaptcha.background.clear.to", "255,255,255");
        }
        return properties;
    }

    public boolean hasShadow() {
        return profile.equals("shadow") || profile.equals("line-shadow");
    }

    public Map<String, String> manifestConfig() {
        Map<String, String> values = new LinkedHashMap<>();
        values.put("width", Integer.toString(width));
        values.put("height", Integer.toString(height));
        values.put("length", Integer.toString(length));
        values.put("charset", charset);
        values.put("format", format);
        values.put("profile", profile);
        return values;
    }
}
