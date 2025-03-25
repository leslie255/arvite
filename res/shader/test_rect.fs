#version 140

in vec2 vert_position;
in vec2 vert_uv;

out vec4 frag_color;

uniform float aaf;
uniform int i_start;
uniform int i_end;
uniform samplerBuffer polygon_points;

float sd_polygon(int i_start, int i_end, out bool clockwise) {
    // Thanks IQ the shader god: https://iquilezles.org/articles/distfunctions2d/
    vec2 p0 = vec2(
            texelFetch(polygon_points, 0).x,
            texelFetch(polygon_points, 1).x);
    float d = dot(vert_uv - p0, vert_uv - p0);
    float s = 1.0;
    int n = i_end - i_start;
    float clockwise_det = 0.0; // determinant for if it's clockwise
    for (int i = i_start, j = i_end - 2; i < n; j = i, i += 2) {
        vec2 v_i = vec2(
            texelFetch(polygon_points, i).x,
            texelFetch(polygon_points, i + 1).x);
        vec2 v_j = vec2(
            texelFetch(polygon_points, j).x,
            texelFetch(polygon_points, j + 1).x);
        clockwise_det += (v_j.x - v_i.x) * (v_j.y + v_i.y);
        vec2 e = v_j - v_i;
        vec2 w = vert_uv - v_i;
        vec2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min( d, dot(b,b) );
        bvec3 c = bvec3(
            vert_uv.y >= v_i.y,
            vert_uv.y < v_j.y,
            e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) s *= -1.0;  
    }
    clockwise = clockwise_det >= 0.0;
    return s * sqrt(d);
}

void main() {
    
    bool clockwise = false;
    float sd = sd_polygon(i_start, i_end, clockwise);

    float aaf_ = fwidth(vert_uv.x) * aaf;
    frag_color = vec4(smoothstep(-aaf_, aaf_, -sd));

    // For debugging:

    // float dx = fwidth(vert_uv.x);
    // if (abs(sd) < dx) {
    //     frag_color = vec4(1.0, 1.0, 1.0, 1.0);
    // } else {
    //     float d_pos = sqrt(clamp(max(sd, 0.0), 0.0, 1.0));
    //     float d_neg = sqrt(clamp(-min(sd, 0.0), 0.0, 1.0));
    //     frag_color = vec4(d_pos, d_pos + d_neg, d_neg, 1.0);
    // }
}
