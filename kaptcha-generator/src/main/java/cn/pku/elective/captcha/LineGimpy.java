package cn.pku.elective.captcha;

import com.google.code.kaptcha.GimpyEngine;
import com.google.code.kaptcha.util.Configurable;

import java.awt.image.BufferedImage;

/** Keeps Kaptcha's rendered text and applies its native curve-noise producer. */
public final class LineGimpy extends Configurable implements GimpyEngine {
    @Override
    public BufferedImage getDistortedImage(BufferedImage image) {
        var noise = getConfig().getNoiseImpl();
        noise.makeNoise(image, 0.1f, 0.1f, 0.25f, 0.25f);
        noise.makeNoise(image, 0.1f, 0.25f, 0.5f, 0.9f);
        return image;
    }
}
