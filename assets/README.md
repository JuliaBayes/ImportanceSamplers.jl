# Logo

`logo.svg` is the source artwork. The transparent PNG exports have square sizes
of 96, 192, 256, 512, and 1024 pixels. The artwork uses the
[Julia palette](https://github.com/JuliaLang/julia-logo-graphics) and the
repository's Apache 2.0 license.

The blue curve is a standard Gaussian proposal, q. The purple target is
π(x) = 0.75 Normal(x; 0.7, 0.48²) + 0.25 Normal(x; −0.4, 1²), where the second
parameter denotes variance. The curves share coordinate scales.

The dots lie at x = [−2.1, −1.25, −0.35, 0.8, 2.1]. Their areas are proportional
to wᵢ = π(xᵢ)/q(xᵢ). Their radii are 16 √(wᵢ/maxⱼ wⱼ), rounded to four decimal
places in the SVG. The largest weight is red. The logo has no frame.

Regenerate a PNG with [librsvg](https://gitlab.gnome.org/GNOME/librsvg):

```sh
rsvg-convert --width 512 --height 512 --output assets/logo-512.png assets/logo.svg
```
