#version 140

in vec2 vert_position;
in vec2 vert_uv;

out vec4 frag_color;

uniform sampler2D tex;

void main() {
    frag_color = texture(tex, vert_uv).r * vec4(1.0, 1.0, 1.0, 1.0);
}
