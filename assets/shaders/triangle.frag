#version 450

layout(location = 0) in vec3 fragColour;
layout(location = 1) in vec2 fragTex;

layout(set = 1, binding = 0) uniform texture2D texture;
layout(set = 1, binding = 1) uniform sampler textureSampler;

layout(location = 0) out vec4 outColour;

void main() {
    outColour = texture(sampler2D(texture, textureSampler), fragTex);
}