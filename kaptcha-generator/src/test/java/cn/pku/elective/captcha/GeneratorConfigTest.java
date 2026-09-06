package cn.pku.elective.captcha;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.*;

class GeneratorConfigTest {
    @Test
    void defaultsMatchTheExistingTrainingContract() {
        GeneratorConfig config = GeneratorConfig.defaults().validate();
        assertEquals(130, config.width());
        assertEquals(52, config.height());
        assertEquals(5, config.length());
        assertEquals("2345678abcdefgmnpwxy", config.charset());
    }

    @Test
    void labelsAreDeterministicForTheSameSeed() {
        String first = KaptchaGenerator.label("abc123", 5, new SplittableRandom(42));
        String second = KaptchaGenerator.label("abc123", 5, new SplittableRandom(42));
        assertEquals(first, second);
        assertEquals(5, first.length());
    }

    @Test
    void invalidConfigurationIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new GeneratorConfig(130, 52, 5, "aab", "png", "default", 1).validate());
        assertThrows(IllegalArgumentException.class,
                () -> new GeneratorConfig(130, 52, 5, "abc", "gif", "default", 1).validate());
    }
}
