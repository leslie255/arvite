#version 140

in vec2 vert_uv;
out vec4 color;

uniform sampler2D tex;
uniform vec4 fg_color;
uniform vec4 bg_color;

void main() {
    color = texture(tex, vert_uv);
    color = color.a * fg_color + (1.0 - color.a) * bg_color;
}
