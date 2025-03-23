#version 140

in vec2 vert_position;
in vec2 vert_uv;

out vec4 frag_color;

uniform sampler2D tex;

void main() {
    float sd = texture(tex, vert_uv).r;

    // A more direct visualization.
    // Yellow is positive.
    // Cyan is negative.
    // float d_pos = sqrt(clamp(max(sd, 0.0), 0.0, 1.0));
    // float d_neg = sqrt(clamp(-min(sd, 0.0), 0.0, 1.0));
    // frag_color = vec4(d_pos, d_pos + d_neg, d_neg, 1.0);

    float aaf = fwidth(vert_uv.x) * 1.0;
    frag_color = vec4(smoothstep(-aaf, aaf, -sd));
}
