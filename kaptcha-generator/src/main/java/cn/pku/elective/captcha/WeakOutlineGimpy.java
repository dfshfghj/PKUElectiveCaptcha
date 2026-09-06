package cn.pku.elective.captcha;

import com.google.code.kaptcha.GimpyEngine;
import com.google.code.kaptcha.impl.WaterRipple;
import com.google.code.kaptcha.util.Configurable;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

/** Adds a one-pixel, low-opacity outline around Kaptcha's rendered glyphs. */
public final class WeakOutlineGimpy extends Configurable implements GimpyEngine {
    @Override
    public BufferedImage getDistortedImage(BufferedImage source) {
        WaterRipple ripple = new WaterRipple();
        ripple.setConfig(getConfig());
        BufferedImage distorted = ripple.getDistortedImage(source);
        BufferedImage result = new BufferedImage(distorted.getWidth(), distorted.getHeight(), BufferedImage.TYPE_INT_ARGB);
        int outline = new Color(20, 24, 30, 120).getRGB();
        for (int y = 0; y < distorted.getHeight(); y++) {
            for (int x = 0; x < distorted.getWidth(); x++) {
                if ((distorted.getRGB(x, y) >>> 24) == 0) continue;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < distorted.getWidth() && ny >= 0 && ny < distorted.getHeight()
                                && (dx != 0 || dy != 0) && (result.getRGB(nx, ny) >>> 24) == 0) {
                            result.setRGB(nx, ny, outline);
                        }
                    }
                }
            }
        }
        Graphics2D graphics = result.createGraphics();
        graphics.drawImage(distorted, 0, 0, null);
        graphics.dispose();
        return result;
    }
}
