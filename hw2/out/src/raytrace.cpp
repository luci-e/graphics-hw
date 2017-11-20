#include <thread>
#include "yocto_img.h"
#include "yocto_math.h"
#include "yocto_scn.h"
#include "yocto_utils.h"
#include <random>
#include <chrono>

#define LINE_CENTROID(a,b) ( ((a) + (b)) / 2.f)
#define TRIANGLE_CENTROID(a,b,c) ( ((a) + (b) + (c)) / 3.f)
#define QUAD_CENTROID(a,b,c,d) ( ((a) + (b) + (c) + (d)) / 4.f)

struct point_info {
	ym::vec3f point;
	ym::vec3f normal;
	ym::vec2f texture_coords;
};

template <typename T>
const void print_vector(T vector, std::string prefix, int size) {
	std::string vec = prefix;
	std::cout << prefix;

	for (int i = 0; i < size; i++) {
		std::cout << " " << vector[i];
	}
}

const void print_image(ym::image4f *img, int n, int m) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			print_vector(img->at({ i,j }), "", 4u);
		}
		std::cout << std::endl;
	}
}

const void print_image(ym::image4b *img, int n, int m) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			print_vector(img->at({ i,j }), "", 4u);
		}
		std::cout << std::endl;
	}
}

ym::ray3f eval_camera(const yscn::camera* cam, const ym::vec2f& uv) {
    auto h = 2 * std::tan(cam->yfov / 2);
    auto w = h * cam->aspect;
    auto q = ym::vec3f{w * cam->focus * (uv.x - 0.5f),
        h * cam->focus * (uv.y - 0.5f), -cam->focus};
    return ym::ray3f(transform_point(cam->frame, ym::zero3f),
        transform_direction(cam->frame, ym::normalize(q - ym::zero3f)));
}

inline ym::vec4f lookup_texture(
    const yscn::texture* txt, const ym::vec2i& ij, bool srgb = true) {
    if (txt->ldr) {
        auto v = txt->ldr[ij];
        return (srgb) ? ym::srgb_to_linear(v) : ym::byte_to_float(v);
    } else if (txt->hdr) {
        return txt->hdr[ij];
    } else {
        assert(false);
        return {};
    }
}

ym::vec4f eval_texture(
    const yscn::texture* txt, const ym::vec2f& texcoord, bool srgb = true) {
    if (!txt) return {1, 1, 1, 1};
    auto wh = ym::vec2i{txt->width(), txt->height()};

    auto st = ym::vec2f{
        std::fmod(texcoord.x, 1.0f) * wh.x, std::fmod(texcoord.y, 1.0f) * wh.y};
    if (st.x < 0) st.x += wh.x;
    if (st.y < 0) st.y += wh.y;

    auto ij = ym::clamp(ym::vec2i{(int)st.x, (int)st.y}, {0, 0}, wh);
    auto uv = st - ym::vec2f{(float)ij.x, (float)ij.y};

    ym::vec2i idx[4] = {ij, {ij.x, (ij.y + 1) % wh.y},
        {(ij.x + 1) % wh.x, ij.y}, {(ij.x + 1) % wh.x, (ij.y + 1) % wh.y}};
    auto w = ym::vec4f{(1 - uv.x) * (1 - uv.y), (1 - uv.x) * uv.y,
        uv.x * (1 - uv.y), uv.x * uv.y};

    // handle interpolation
    return (lookup_texture(txt, idx[0], srgb) * w.x +
            lookup_texture(txt, idx[1], srgb) * w.y +
            lookup_texture(txt, idx[2], srgb) * w.z +
            lookup_texture(txt, idx[3], srgb) * w.w);
}

ym::vec3f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec3f> vecs, int vec_no) {
	ym::vec3f vec = { 0, 0, 0 };
	for (int i = 0; i < vec_no; i++) {
		vec += bar_coord[i] * vecs[i];
	}
	return vec;
}

inline ym::vec3f triangle_barycentric_to_vec(ym::vec2f uv, std::array<ym::vec3f, 3u> vertices) {
	return vertices[0] + uv.x * (vertices[1] - vertices[0]) + uv.y*(vertices[2] - vertices[0]);
}

inline ym::vec2f triangle_barycentric_to_vec(ym::vec2f uv, std::array<ym::vec2f, 3u> vertices) {
	return vertices[0] + uv.x * (vertices[1] - vertices[0]) + uv.y*(vertices[2] - vertices[0]);
}

ym::vec2f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec2f> vecs, int vec_no) {
	ym::vec2f vec = { 0, 0 };
	for (int i = 0; i < vec_no; i++) {
		vec += bar_coord[i] * vecs[i];
	}
	return vec;
}

// Compute the opaqueness at the intersection point
float opaqueness_at_intersection(const yscn::scene* scn, const yscn::intersection_point intersection) {
	float op;
	auto instance = scn->instances[intersection.iid];
	auto shape = instance->shp;
	auto mat = shape->mat;

	op = mat->op;

	if (shape->texcoord.size() > 0 && mat->occ_txt.txt) {
		ym::vec2f texture_coords;
		if (shape->points.size() != 0) {
			auto p = shape->points[intersection.eid];
			texture_coords = shape->texcoord[p];
		}
		else if (shape->lines.size() != 0) {
			auto l = shape->lines[intersection.eid];
			std::vector<ym::vec2f> textures = { shape->texcoord[l[0]], shape->texcoord[l[1]] };
			texture_coords = barycentric_to_vec(intersection.euv, textures, 2);
		}
		else if (shape->triangles.size() != 0) {
			auto t = shape->triangles[intersection.eid];
			std::vector<ym::vec2f> textures = { shape->texcoord[t[0]], shape->texcoord[t[1]], shape->texcoord[t[2]] };
			texture_coords = barycentric_to_vec(intersection.euv, textures, 3);
		}

		op *= eval_texture(mat->occ_txt.txt, texture_coords, false)[0];
	}

	return op;
}

point_info get_point_info(const yscn::scene* scn, const yscn::intersection_point intersection, bool is_textured = false) {
	point_info p_i;
	auto shape = scn->instances[intersection.iid]->shp;

	// Determine the type of shape
	if (shape->points.size() != 0) {
		auto p = shape->points[intersection.eid];
		p_i.point = shape->pos[p];
		p_i.normal = shape->norm[p];
		if (is_textured) {
			p_i.texture_coords = shape->texcoord[p];
		}
	}
	else if (shape->lines.size() != 0) {
		auto l = shape->lines[intersection.eid];
		std::vector<ym::vec3f> vectors = { shape->pos[l[0]], shape->pos[l[1]] };
		std::vector<ym::vec3f> normals = { shape->norm[l[0]], shape->norm[l[1]] };

		// Goodbye simmetry hello efficiency
		if (is_textured) {
			std::vector<ym::vec2f> textures = { shape->texcoord[l[0]], shape->texcoord[l[1]] };
			p_i.texture_coords = barycentric_to_vec(intersection.euv, textures, 2);
		}

		p_i.point = barycentric_to_vec(intersection.euv, vectors, 2);
		p_i.normal = ym::normalize(barycentric_to_vec(intersection.euv, normals, 2));
	}
	else if (shape->triangles.size() != 0) {
		auto t = shape->triangles[intersection.eid];
		std::vector<ym::vec3f> vectors = { shape->pos[t[0]], shape->pos[t[1]], shape->pos[t[2]] };
		std::vector<ym::vec3f> normals = { shape->norm[t[0]], shape->norm[t[1]], shape->norm[t[2]] };

		// Goodbye simmetry hello efficiency
		if (is_textured) {
			std::vector<ym::vec2f> textures = { shape->texcoord[t[0]], shape->texcoord[t[1]], shape->texcoord[t[2]] };
			p_i.texture_coords = barycentric_to_vec(intersection.euv, textures, 3);
		}

		p_i.point = barycentric_to_vec(intersection.euv, vectors, 3);
		p_i.normal = ym::normalize(barycentric_to_vec(intersection.euv, normals, 3));
	}

	return p_i;
};

ym::vec4f shade_light(const yscn::scene* scn, const yscn::instance* light, const int light_point, const ym::ray3f& ray, bool coloured_light = false) {
	ym::vec3f light_colour;

	auto light_shape = light->shp;
	auto light_mat = light_shape->mat;
	auto light_transform = light->xform();

	auto light_pos = light_shape->pos[light_point];
	// Bring back to world coord
	light_pos = ym::transform_point(light_transform, light_pos);

	auto light_direction = ym::normalize(light_pos - ray.o);
	auto light_distance = ym::length(light_pos - ray.o);

	auto light_intersection = yscn::intersect_ray(scn, ym::ray3f(ray.o, ray.d, 1e-6f, light_distance), false);

	// Dafuq did we hit?
	if (coloured_light) {
		if (light_intersection && !(scn->instances[light_intersection.iid] == light)) {
			auto light_hit_instance = scn->instances[light_intersection.iid];
			float light_op;
			if ((light_op = opaqueness_at_intersection(scn, light_intersection)) < 1) {
				auto mat = light_hit_instance->shp->mat;
				auto kd = mat->kd;
				bool is_textured = (mat->kd_txt) ? true : false;

				auto light_hit_point_i = get_point_info(scn, light_intersection, is_textured);
				auto light_hit_point = light_hit_point_i.point;

				if (is_textured) {
					auto kd_4 = eval_texture(mat->kd_txt.txt, light_hit_point_i.texture_coords, true);
					kd *= { kd_4[0], kd_4[1], kd_4[2] };
				}

				light_hit_point = ym::transform_point(light_hit_instance->xform(), light_hit_point);
				auto light_hit_colour = shade_light(scn, light, light_point, ym::ray3f(light_hit_point, light_direction, 1e-6f));
				if (light_hit_colour[3] < 1.f) {
					return (1.f - light_op) * ym::vec4f{ kd[0] * light_hit_colour[0], kd[1] * light_hit_colour[1], kd[2] * light_hit_colour[2], 1.f };
				}
			}
			// President Shader, an opaque body is blocking our lights in Brazil
			// SHUT. DOWN. EVERYTHING!
			return { 0.f, 0.f, 0.f, 1.f };
		}
	}

	// If we intersect a light we can finally return a light intensity!
	if (!light_intersection || (scn->instances[light_intersection.iid] == light)) {
		return { light_mat->ke[0], light_mat->ke[1], light_mat->ke[2], 0.f };
	}

	return { 0.f, 0.f, 0.f, 1.f };
}


ym::vec4f shade(const yscn::scene* scn,
				const std::vector<yscn::instance*>& lights, const ym::vec3f& amb,
				const ym::ray3f& ray, float current_light = 1.f) {
	auto isec = yscn::intersect_ray(scn, ray, false);

	if (current_light < 0.3) return { 0, 0, 0, 0 };
	if (!isec) return { 0, 0, 0, 0 };

	auto ist = scn->instances[isec.iid];
	auto shp = ist->shp;
	auto mat = shp->mat;

	auto pos = ym::zero3f;
	auto norm = ym::zero3f;
	auto texcoord = ym::zero2f;
	if (!shp->triangles.empty()) {
		auto t = shp->triangles[isec.eid];
		pos = ym::eval_barycentric_triangle(
			shp->pos[t.x], shp->pos[t.y], shp->pos[t.z], isec.euv);
		norm = ym::normalize(ym::eval_barycentric_triangle(
			shp->norm[t.x], shp->norm[t.y], shp->norm[t.z], isec.euv));
		if (!shp->texcoord.empty())
			texcoord = ym::eval_barycentric_triangle(shp->texcoord[t.x],
													 shp->texcoord[t.y], shp->texcoord[t.z], isec.euv);
	}
	else if (!shp->lines.empty()) {
		auto l = shp->lines[isec.eid];
		pos = ym::eval_barycentric_line(shp->pos[l.x], shp->pos[l.y], isec.euv);
		norm = ym::normalize(ym::eval_barycentric_line(
			shp->norm[l.x], shp->norm[l.y], isec.euv));
		if (!shp->texcoord.empty())
			texcoord = ym::eval_barycentric_line(
			shp->texcoord[l.x], shp->texcoord[l.y], isec.euv);
	}
	else if (!shp->points.empty()) {
		auto p = shp->points[isec.eid];
		pos = shp->pos[p];
		norm = shp->norm[p];
		if (!shp->texcoord.empty()) texcoord = shp->texcoord[p];
	}

	pos = ym::transform_point(ym::to_frame(ist->xform()), pos);
	norm = ym::transform_direction(ym::to_frame(ist->xform()), norm);

	auto ke = mat->ke * eval_texture(mat->ke_txt.txt, texcoord).xyz();
	auto kd = mat->kd * eval_texture(mat->kd_txt.txt, texcoord).xyz();
	auto ks = mat->ks * eval_texture(mat->ks_txt.txt, texcoord).xyz();
	auto ns = (mat->rs) ? 2 / (mat->rs * mat->rs) - 2 : 1e6f;

	auto l = ke + kd * amb;
	for (auto lgt : lights) {
		auto lshp = lgt->shp;
		for (auto p : lgt->shp->points) {
			auto lpos =
				ym::transform_point(ym::to_frame(lgt->xform()), lshp->pos[p]);
			auto lr = ym::length(lpos - pos);
			auto wi = ym::normalize(lpos - pos);

			auto light = shade_light(scn, lgt, p, { pos, wi, 0.01f, lr - 0.04f }, true);
			
			if (light[3] != 1.f) {
				auto le = light.xyz() / (lr * lr * ym::pif);  // normalized to pi, why?

				auto wo = -ray.d;
				auto wh = ym::normalize(wi + wo);
				if (!shp->triangles.empty()) {
					l += le * kd * ym::max(0.0f, ym::dot(norm, wi)) +
						le * ks * ym::pow(ym::max(0.0f, ym::dot(norm, wh)), ns);
				}
				else if (!shp->lines.empty()) {
					l += le * kd *
						ym::sqrt(ym::clamp(
						1 - ym::dot(norm, wi) * ym::dot(norm, wi), 0.0f,
						1.0f)) +
						le * ks *
						ym::pow(ym::sqrt(ym::clamp(
						1 - ym::dot(norm, wh) * ym::dot(norm, wh),
						0.0f, 1.0f)),
						ns);
				}
			}
		}
	}

	// Handle opacity for hairs
	ym::vec3f colour = l;
	if (mat->op < 1.f) {
		colour = mat->op*l + (1.f - mat->op)*shade(scn, lights, amb, ym::ray3f(pos, ray.d, 1e-6f), (current_light) * (1 - mat->op)).xyz();
	}

	return { colour.x, colour.y, colour.z, 1 };
}

ym::vec4f shade_2(const yscn::scene* scn,
    const std::vector<yscn::instance*>& lights, const ym::vec3f& amb,
    const ym::ray3f& ray, float current_light = 1.f) {
    auto isec = yscn::intersect_ray(scn, ray, false);

	if (current_light < 0.3) return { 0, 0, 0, 0 };
    if (!isec) return {0, 0, 0, 0};

    auto ist = scn->instances[isec.iid];
    auto shp = ist->shp;
    auto mat = shp->mat;

    auto pos = ym::zero3f;
    auto norm = ym::zero3f;
    auto texcoord = ym::zero2f;
    if (!shp->triangles.empty()) {
        auto t = shp->triangles[isec.eid];
        pos = ym::eval_barycentric_triangle(
            shp->pos[t.x], shp->pos[t.y], shp->pos[t.z], isec.euv);
        norm = ym::normalize(ym::eval_barycentric_triangle(
            shp->norm[t.x], shp->norm[t.y], shp->norm[t.z], isec.euv));
        if (!shp->texcoord.empty())
            texcoord = ym::eval_barycentric_triangle(shp->texcoord[t.x],
                shp->texcoord[t.y], shp->texcoord[t.z], isec.euv);
    } else if (!shp->lines.empty()) {
        auto l = shp->lines[isec.eid];
        pos = ym::eval_barycentric_line(shp->pos[l.x], shp->pos[l.y], isec.euv);
        norm = ym::normalize(ym::eval_barycentric_line(
            shp->norm[l.x], shp->norm[l.y], isec.euv));
        if (!shp->texcoord.empty())
            texcoord = ym::eval_barycentric_line(
                shp->texcoord[l.x], shp->texcoord[l.y], isec.euv);
    } else if (!shp->points.empty()) {
        auto p = shp->points[isec.eid];
        pos = shp->pos[p];
        norm = shp->norm[p];
        if (!shp->texcoord.empty()) texcoord = shp->texcoord[p];
    }

    pos = ym::transform_point(ym::to_frame(ist->xform()), pos);
    norm = ym::transform_direction(ym::to_frame(ist->xform()), norm);

    auto ke = mat->ke * eval_texture(mat->ke_txt.txt, texcoord).xyz();
    auto kd = mat->kd * eval_texture(mat->kd_txt.txt, texcoord).xyz();
    auto ks = mat->ks * eval_texture(mat->ks_txt.txt, texcoord).xyz();
    auto ns = (mat->rs) ? 2 / (mat->rs * mat->rs) - 2 : 1e6f;

    auto l = ke + kd * amb;
    for (auto lgt : lights) {
        auto lshp = lgt->shp;
        for (auto p : lgt->shp->points) {
            auto lpos =
                ym::transform_point(ym::to_frame(lgt->xform()), lshp->pos[p]);
            auto lr = ym::length(lpos - pos);
            auto wi = ym::normalize(lpos - pos);
            if (yscn::intersect_ray(scn, {pos, wi, 0.01f, lr - 0.04f}, true))
                continue;

            auto le = lshp->mat->ke / (lr * lr * ym::pif);  // normalized to pi
            auto wo = -ray.d;
            auto wh = ym::normalize(wi + wo);
            if (!shp->triangles.empty()) {
                l += le * kd * ym::max(0.0f, ym::dot(norm, wi)) +
                     le * ks * ym::pow(ym::max(0.0f, ym::dot(norm, wh)), ns);
            } else if (!shp->lines.empty()) {
                l += le * kd *
                         ym::sqrt(ym::clamp(
                             1 - ym::dot(norm, wi) * ym::dot(norm, wi), 0.0f,
                             1.0f)) +
                     le * ks *
                         ym::pow(ym::sqrt(ym::clamp(
                                     1 - ym::dot(norm, wh) * ym::dot(norm, wh),
                                     0.0f, 1.0f)),
                             ns);
            }
        }
    }

	// Handle opacity for hairs
	ym::vec3f colour = l;
	if (mat->op < 1.f) {
		colour = mat->op*l + (1.f - mat->op)*shade(scn, lights, amb, ym::ray3f(pos, ray.d, 1e-6f), (current_light) * (1- mat->op)).xyz();
	}

    return { colour.x, colour.y, colour.z, 1};
}

ym::image4f raytrace(const yscn::scene* scn, const ym::vec3f& amb,
    int resolution, int samples, bool facets) {
    auto cam = scn->cameras.front();
    auto img = ym::image4f(
        (int)std::round(cam->aspect * resolution), resolution, {0, 0, 0, 0});

    auto lights = std::vector<yscn::instance*>();
    for (auto ist : scn->instances) {
        if (ist->shp->mat->ke == ym::zero3f) continue;
        if (ist->shp->points.empty()) continue;
        lights.push_back(ist);
    }

    for (auto j = 0; j < img.height(); j++) {
        for (auto i = 0; i < img.width(); i++) {
            img[{i, j}] = {0, 0, 0, 0};
            for (auto sj = 0; sj < samples; sj++) {
                for (auto si = 0; si < samples; si++) {
                    auto u = (i + (si + 0.5f) / samples) / img.width();
                    auto v = ((img.height() - j) + (sj + 0.5f) / samples) /
                             img.height();
                    auto ray = eval_camera(cam, {u, v});
                    img.at(i, j) += shade(scn, lights, amb, ray);
                }
            }
            img[{i, j}] /= (float)(samples * samples);
        }
    }

    return img;
}


void timer_start(std::atomic_ulong &counter, unsigned long total, unsigned int interval) {
	std::thread([&counter, total, interval]() {
		while (true) {
			std::cout << "Completion: " << (float)counter / (float)total<< std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(interval));
		}
	}).detach();
}


ym::image4f raytrace_mt(const yscn::scene* scn, const ym::vec3f& amb,
    int resolution, int samples, bool facets) {
    auto cam = scn->cameras.front();
    auto img = ym::image4f(
        (int)std::round(cam->aspect * resolution), resolution, {0, 0, 0, 0});

    auto lights = std::vector<yscn::instance*>();
    for (auto ist : scn->instances) {
        if (ist->shp->mat->ke == ym::zero3f) continue;
        if (ist->shp->points.empty()) continue;
        lights.push_back(ist);
    }

	// TODO
	std::atomic_ulong done_pixels = 0;
	unsigned long total_pixels = (unsigned long)(img.height() * img.width());

	timer_start(done_pixels, total_pixels, 1000);

    auto nthreads = std::thread::hardware_concurrency();
    auto threads = std::vector<std::thread>();
    for (auto tid = 0; tid < nthreads; tid++) {
        threads.push_back(std::thread([=, &img, &done_pixels]() {
            for (auto j = tid; j < img.height(); j += nthreads) {
                for (auto i = 0; i < img.width(); i++) {
                    img[{i, j}] = {0, 0, 0, 0};
                    for (auto sj = 0; sj < samples; sj++) {
                        for (auto si = 0; si < samples; si++) {
                            auto u = (i + (si + 0.5f) / samples) / img.width();
                            auto v =
                                ((img.height() - j) + (sj + 0.5f) / samples) /
                                img.height();
                            auto ray = eval_camera(cam, {u, v});
                            img.at(i, j) += shade(scn, lights, amb, ray);
                        }
                    }
	
					done_pixels++;
                    img[{i, j}] /= (float)(samples * samples);
                }
            }
        }));
    }


    for (auto& thread : threads) thread.join();
    return img;
}


// Compute the centroid of a polygon. This is really slow, use macros for simple polygons!
template <size_t T>
inline ym::vec3f centroid(std::array<ym::vec3f, T> vertices) {
	ym::vec3f centroid;
	for (size_t i = 0; i < T; i++) { centroid += vertices[i]; }
	return centroid / (float) T;
}

//
// Displace each vertex of the shape along the normal of a distance
// <texture_value> * scale. After displacement, compute smooth normals
// with `ym::compute_normals()`.
//
void displace(yscn::shape* shp, float scale) {
	if (!shp->quads.empty() || !shp->triangles.empty()) {
		// Hey you gotta have a texture
		assert(shp->mat->disp_txt != NULL);

		for (int i = 0; i < shp->norm.size(); i++) {
			auto disp = eval_texture(shp->mat->disp_txt.txt, shp->texcoord[i]);
			disp *= scale;

			shp->pos[i] += (shp->norm[i] * disp[0]);
		}

		if (!shp->quads.empty()) { ym::compute_normals(shp->quads, shp->pos, shp->norm); }
		else if (!shp->triangles.empty()) { ym::compute_normals(shp->triangles, shp->pos, shp->norm); }
	} else { yu::logging::log_error("Can only displace quads or triangles!"); }
}

// Tessellate triangle mesh with Loop algorithm
void tessellate_triangle_mesh(yscn::shape* shp) {
	auto e_map = ym::edge_map{ shp->triangles };

	auto edge_no = e_map.get_edges().size();
	auto pos_no = shp->pos.size();
	auto triangles_no = shp->triangles.size();

	bool txc_f = !shp->texcoord.empty();
	bool txc1_f = !shp->texcoord1.empty();
	bool clr_f = !shp->color.empty();
	bool tgs_f = !shp->tangsp.empty();

	// Expand the shape vectors to accomodate for the new points
	shp->pos.resize(pos_no + edge_no);
	shp->norm.resize(pos_no + edge_no);
	if (txc_f) { shp->texcoord.resize(pos_no + edge_no); }
	if (txc1_f) { shp->texcoord1.resize(pos_no + edge_no); }
	if (clr_f) { shp->color.resize(pos_no + edge_no); }
	if (tgs_f) { shp->tangsp.resize(pos_no + edge_no); }

	// Add them at starting from the last
	for (auto edge : e_map.get_edges()) {
		auto new_no = pos_no + e_map.at(edge);
		
		// Compute new values as centroid of line
		auto e_pos = LINE_CENTROID(shp->pos[edge.x], shp->pos[edge.y]);
		auto e_nor = LINE_CENTROID(shp->norm[edge.x], shp->norm[edge.y]);
		if (txc_f) { auto e_txc = LINE_CENTROID(shp->texcoord[edge.x], shp->texcoord[edge.y]); shp->texcoord[new_no] = e_txc;}
		if (txc1_f) { auto e_txc1 = LINE_CENTROID(shp->texcoord1[edge.x], shp->texcoord1[edge.y]); shp->texcoord1[new_no] = e_txc1; }
		if (clr_f) { auto e_col = LINE_CENTROID(shp->color[edge.x], shp->color[edge.y]); shp->color[new_no] = e_col; }

		// Add them to the shape
		shp->pos[new_no] = e_pos;
		shp->norm[new_no] = e_nor;
	}

	// Create the new triangles
	for (int i = 0; i < triangles_no;  i++) {
		auto triangle = shp->triangles[i];

		// Get the points of the triangles
		auto v0 = triangle.x;
		auto v1 = triangle.y;
		auto v2 = triangle.z;

		// Get the new edge-points numbers, in order
		int e0 = pos_no + e_map.at({v0, v1});
		int e1 = pos_no + e_map.at({v1, v2});
		int e2 = pos_no + e_map.at({v2, v0});

		// Add the new triangles, change the first
		shp->triangles[i] = { v0, e0, e2 };
		shp->triangles.push_back({ v1, e1, e0 });
		shp->triangles.push_back({ v2, e2, e1 });
		shp->triangles.push_back({ e0, e1, e2 });
	}
}

// Tessellate quad mesh with Catmull-Clark algorithm
void tessellate_quad_mesh(yscn::shape* shp) {
	auto e_map = ym::edge_map{ shp->quads };

	auto edge_no = e_map.get_edges().size();
	auto pos_no = shp->pos.size();
	auto quads_no = shp->quads.size();

	bool txc_f = !shp->texcoord.empty();
	bool txc1_f = !shp->texcoord1.empty();
	bool clr_f = !shp->color.empty();
	bool tgs_f = !shp->tangsp.empty();


	// Expand the shape vectors to accomodate for the new points
	shp->pos.resize(pos_no + edge_no + quads_no);
	shp->norm.resize(pos_no + edge_no + quads_no);
	if (txc_f) { shp->texcoord.resize(pos_no + edge_no + quads_no); }
	if (txc1_f) { shp->texcoord1.resize(pos_no + edge_no + quads_no); }
	if (clr_f) { shp->color.resize(pos_no + edge_no + quads_no); }
	if (tgs_f) { shp->tangsp.resize(pos_no + edge_no + quads_no); }

	// Append the new values to the vectors
	for (auto edge : e_map.get_edges()) {
		auto new_no = pos_no + e_map.at(edge);

		// Compute new values as centroid of line
		auto e_pos = LINE_CENTROID(shp->pos[edge.x], shp->pos[edge.y]);
		auto e_nor = LINE_CENTROID(shp->norm[edge.x], shp->norm[edge.y]);
		if (txc_f) { auto e_txc = LINE_CENTROID(shp->texcoord[edge.x], shp->texcoord[edge.y]); shp->texcoord[new_no] = e_txc; }
		if (txc1_f) { auto e_txc1 = LINE_CENTROID(shp->texcoord1[edge.x], shp->texcoord1[edge.y]); shp->texcoord1[new_no] = e_txc1; }
		if (clr_f) { auto e_col = LINE_CENTROID(shp->color[edge.x], shp->color[edge.y]); shp->color[new_no] = e_col; }

		// Add them to the shape
		shp->pos[new_no] = e_pos;
		shp->norm[new_no] = e_nor;
	}

	// Create the new quads
	for (int i = 0; i < quads_no; i++) {
		auto quad = shp->quads[i];

		// Get the points of the quad
		auto v0 = quad.x;
		auto v1 = quad.y;
		auto v2 = quad.z;
		auto v3 = quad.w;

		// Get the new edge-points numbers, in order
		int e0 = pos_no + e_map.at(ym::vec2i{ v0, v1 });
		int e1 = pos_no + e_map.at(ym::vec2i{ v1, v2 });

		// Handle degenerate quad
		int e2, e3;
		if (v2 != v3) {
			e2 = pos_no + e_map.at(ym::vec2i{ v2, v3 });
			e3 = pos_no + e_map.at(ym::vec2i{ v3, v0 });
		}
		else {
			e2 = pos_no + e_map.at(ym::vec2i{ v2, v0 });
			e3 = e2;
		}
		int e5 = pos_no + edge_no + i;

		// Compute new values as centroid of quad
		auto e_pos = QUAD_CENTROID(shp->pos[v0], shp->pos[v1], shp->pos[v2], shp->pos[v3]);
		auto e_nor = QUAD_CENTROID(shp->norm[v0], shp->norm[e1], shp->norm[v2], shp->norm[v3]);
		if (txc_f) { auto e_txc = QUAD_CENTROID(shp->texcoord[v0], shp->texcoord[v1], shp->texcoord[v2], shp->texcoord[v3]); shp->texcoord[e5] = e_txc; }
		if (txc1_f) { auto e_txc1 = QUAD_CENTROID(shp->texcoord1[v0], shp->texcoord1[v1], shp->texcoord1[v2], shp->texcoord1[v3]); shp->texcoord1[e5] = e_txc1; }
		if (clr_f) { auto e_col = QUAD_CENTROID(shp->color[v0], shp->color[v1], shp->color[v2], shp->color[v3]); shp->color[e5] = e_col; }

		// Add them to the shape
		shp->pos[e5] = e_pos;
		shp->norm[e5] = e_nor;

		// Add the new quads, change the first
		shp->quads[i] = { v0, e0, e5, e3 };
		shp->quads.push_back({ v1, e1, e5, e0 });
		shp->quads.push_back({ v2, e2, e5, e1 });
		if (v2 != v3) {
			shp->quads.push_back({ v3, e3, e5, e2 });
		}
	}
}

//
// Linear tesselation that split each triangle in 4 triangles and each quad in
// four quad. Vertices are placed in the middle of edges and faces. See slides
// on subdivision for the geometry.
// The tesselation operation should be repeated `level` times.
// To simplify coding, shapes are either quads ot triangles (depending on which
// array is filled). In the case of quad meshes, we include triangles by
// specifying a degenerate quad that has the last vertex index duplicated as
// v0, v1, v2, v2.
// Implement a different algorithm for quad meshes and triangle meshes.
//
void tesselate(yscn::shape* shp, int level) {
	int i = 0;
	if (!shp->triangles.empty()) {
		while (i++ < level) { tessellate_triangle_mesh(shp); }
		return;
	}
	else if (!shp->quads.empty()) {
		while (i++ < level) { tessellate_quad_mesh(shp); }
		return;
	}

	yu::logging::log_error("Sorry mate, only triangles and quads\n");
}

void smooth_quad_mesh(yscn::shape *shp) {
	auto original_mesh = shp->pos;
	auto avg_v = std::vector < ym::vec3f > (shp->pos.size());
	auto avg_n = std::vector<int>(shp->pos.size(), 0);

	for (auto quad : shp->quads) {
		// Get the points of the quad
		auto v0 = quad.x;
		auto v1 = quad.y;
		auto v2 = quad.z;
		auto v3 = quad.w;

		auto c = QUAD_CENTROID(shp->pos[v0], shp->pos[v1], shp->pos[v2], shp->pos[v3]);

		for (auto v : { v0, v1, v2, v3 }) {
			avg_v[v] += c;
			avg_n[v] += 1;
		}
	}

	for (size_t v_index = 0; v_index < avg_n.size(); v_index++) {
		avg_v[v_index] /= (float) avg_n[v_index];
	}

	for (size_t v_index = 0; v_index < avg_v.size(); v_index++) {
		shp->pos[v_index] += (avg_v[v_index] - original_mesh[v_index]) * (4.f / (float) avg_n[v_index]);
	}
}

void convert_triangles_to_quads(yscn::shape *shp) {
	yu::logging::log_info("Converting triangles to quads for Catmull-Clark");
	shp->quads.resize(shp->triangles.size());
	for (size_t i = 0; i < shp->triangles.size(); i++) {
		auto tri = shp->triangles[i];
		shp->quads[i] = { tri.x, tri.y, tri.z, tri.z };
	}
	shp->triangles.clear();
}

//
// Implement Catmull-Clark subdivision with the algorithm specified in class.
// You should only handle quad meshes, but note that some quad might be
// degenerate and in facgt represent a triangle as described above.
// The whole subvision should be repeated `level` timers.
// At the end, smooth the normals with `ym::compute_normals()`.
//
void catmull_clark(yscn::shape* shp, int level) {
	int i = 0;
	// Someone passed the wrong mesh
	if (!shp->triangles.empty()) { convert_triangles_to_quads(shp); }
	while (i++ < level) {
		tesselate(shp, 1);
		smooth_quad_mesh(shp);
	}
	ym::compute_normals(shp->quads, shp->pos, shp->norm);
}

//
// Add nhair to surfaces by creating a hair shape, made of lines, that is
// sampled over a shape `shp`. Each hair is a line of one segment that starts
// at a location on the surface and grows along the normals to a distance
// `length`. Each hair has the surface normal as it norm and the surface
// texcoord as its texcoord. Use the function `ym::sample_triangles_points()`
// with a seed of 0 to generate the surface pos, norm and texcoord.
//
yscn::shape* make_hair(
    const yscn::shape* shp, int nhair, float length, float radius) {
    auto hair = new yscn::shape();

	// Make space for the hairification
	hair->lines.resize(nhair);

	// Set one radius for all
	hair->radius.resize(2 * nhair);
	std::fill_n(hair->radius.begin(), (2 * nhair), radius);

	ym::sample_triangles_points(shp->triangles, shp->pos, shp->norm, shp->texcoord, nhair, hair->pos, hair->norm, hair->texcoord, 0u);

	// Expand the vectors to double the number of hairs
	hair->pos.resize(2 * nhair);
	hair->norm.resize(2 * nhair);
	hair->texcoord.resize(2 * nhair);

	std::copy_n(hair->norm.begin(), nhair, hair->norm.begin() + nhair);
	std::copy_n(hair->texcoord.begin(), nhair, hair->texcoord.begin() + nhair);

	for (size_t i = 0; i < nhair; i++) {
		hair->pos[i + nhair] = hair->pos[i] + hair->norm[i] * length;
		hair->lines[i] = { (int) i, ((int) i ) + nhair };
	}

    return hair;
}

inline std::array<ym::vec3f, 4> make_rnd_curve(
    const ym::vec3f& pos, const ym::vec3f& norm) {
    // HACK: this is a cheese hack that works
    auto rng = ym::init_rng_pcg32(*(uint32_t*)&pos.x, *(uint32_t*)&pos.y);
    auto v0 = pos;
    auto v1 = v0 + norm * 0.1f +
              (ym::next_rand3f(rng) - ym::vec3f{0.5f, 0.5f, 0.5f}) * 0.05f;
    auto v2 = v1 + norm * 0.1f +
              (ym::next_rand3f(rng) - ym::vec3f{0.5f, 0.5f, 0.5f}) * 0.05f;
    auto v3 = v2 + norm * 0.1f +
              (ym::next_rand3f(rng) - ym::vec3f{0.5f, 0.5f, 0.5f}) * 0.05f;
    return {{v0, v1, v2, v3}};
}


// A struct for holding information about a triangle mesh surface
struct triangle_surface {
	std::vector<float> triangle_areas;
	float total_area = 0.f;

	triangle_surface(size_t triangles) {
		triangle_areas.resize(triangles);
	}

	inline const float operator[](size_t i) {
		return triangle_areas[i];
	}
};

inline void get_triangles_areas(
	int ntriangles, const std::vector<ym::vec3i>& triangles, const std::vector<ym::vec3f>& pos, triangle_surface & tri_a) {
	for (auto i = 0; i < ntriangles; i++) {
		tri_a.triangle_areas[i] = ym::triangle_area(
			pos[triangles[i].x], pos[triangles[i].y], pos[triangles[i].z]);
		tri_a.total_area += tri_a.triangle_areas[i];
	}
}

inline size_t generate_points_on_triangle(const yscn::shape *shp, const std::vector<ym::vec3i>& triangles,
								   const std::vector<ym::vec3f>& pos, const std::vector<ym::vec3f>& norm,
								   const std::vector<ym::vec2f>& texcoord,
								   std::vector<ym::vec3f>& sampled_pos, std::vector<ym::vec3f>& sampled_norm,
								   std::vector<ym::vec2f>& sampled_texcoord, uint64_t seed, size_t triangle_no, triangle_surface &surfaces, ym::rng_pcg32 &rng) {
	// The current triangle
	auto &triangle = triangles[triangle_no];

	// Eval the centroid of the texture, multiply the density value of the texture by the area of the triangle
	// to get the expected value of hairs in the triangle, multiply by a constant cause density only goes up 
	// to 255 hairs / unit squared which is clearly too small
	auto tex_centroid = TRIANGLE_CENTROID(eval_texture(shp->mat->density_txt.txt, texcoord[triangle[0]], false),
										  eval_texture(shp->mat->density_txt.txt, texcoord[triangle[1]], false),
										  eval_texture(shp->mat->density_txt.txt, texcoord[triangle[2]], false));

	auto expected = tex_centroid.x * 8e6  / 255.f * surfaces[triangle_no];

	size_t generated_hair_no = 0u;

	// Compute two barycentric coordinates by generating two floats, summing the second to the first and taking the 1.f modulus
	for (auto i = 0.f; i < expected; i++) {
		auto diff = (expected - i);

		// If there should be less than 1 hair, flip a coin to decide whether to put it or not
		if (diff < 1.f) {
			auto put_hair = ym::next_rand1f(rng) <= diff;
			if (!put_hair) { continue; };
		}

		generated_hair_no++;

		auto uv = ym::next_rand2f(rng);
		uv.y = std::fmodf(uv.x + uv.y, 1.f);

		//Fill the values
		sampled_pos.push_back(triangle_barycentric_to_vec(uv, { pos[triangle[0]], pos[triangle[1]], pos[triangle[2]] }));
		sampled_norm.push_back(triangle_barycentric_to_vec(uv, { norm[triangle[0]], norm[triangle[1]], norm[triangle[2]] }));
		sampled_texcoord.push_back(triangle_barycentric_to_vec(uv, { texcoord[triangle[0]], texcoord[triangle[1]], texcoord[triangle[2]] }));
	}

	return generated_hair_no;
}

// Sample points from density map, returns the number of hair generated
inline size_t sample_points_with_density_map(const yscn::shape *shp, const std::vector<ym::vec3i>& triangles,
							   const std::vector<ym::vec3f>& pos, const std::vector<ym::vec3f>& norm,
							   const std::vector<ym::vec2f>& texcoord,
							   std::vector<ym::vec3f>& sampled_pos, std::vector<ym::vec3f>& sampled_norm,
							   std::vector<ym::vec2f>& sampled_texcoord, uint64_t seed) {

	// Get the areas of the triangles to compute how many points each triangle get
	auto surfaces = triangle_surface(triangles.size());
	get_triangles_areas(triangles.size(), triangles, pos, surfaces);

	auto rng = ym::init_rng_pcg32(seed, 0u);

	size_t total_hairs = 0;

	// Loop over the triangles, generate the points
	for (size_t i = 0; i < triangles.size(); i++) {
		total_hairs += generate_points_on_triangle(shp, triangles, pos, norm, texcoord, sampled_pos, sampled_norm, sampled_texcoord, seed, i, surfaces, rng);
	}

	return total_hairs;
}

struct hair_model {
	float avg_len;
	float avg_radius;
	float elastic_modulus;
	float density;

	float len_var;
	float rad_var;

	std::default_random_engine generator;
	std::normal_distribution<float> len_distribution;
	std::normal_distribution<float> rad_distribution;

	ym::vec4f new_hair() {
		auto h_length = avg_len + len_distribution(generator);
		auto h_radius = avg_radius + rad_distribution(generator);
		auto second_moment_inertia = ym::pif * ym::pow(h_radius, 4) / 4.f;
		auto m_per_unit_len = ym::pif * (h_radius * h_radius) * density;

		return { h_length, h_radius, second_moment_inertia, m_per_unit_len };
	}

	hair_model(float avg_len, float avg_radius, float elastic_modulus, float len_var, float rad_var, float density) : avg_len(avg_len), avg_radius(avg_radius),
	elastic_modulus(elastic_modulus), len_var(len_var), rad_var(rad_var), density(density){
		this->len_distribution = std::normal_distribution<float>(0.f, len_var);
		this->rad_distribution = std::normal_distribution<float>(0.f, rad_var);
	};
};

inline ym::vec3f eval_deflection(ym::vec4f hair_info, float elastic_modulus, ym::vec3f acc_vector, ym::vec3f hair_direction, int nsegs, int current_segment) {
	ym::vec3f deflection;
	auto seg_len = hair_info.x / (float)nsegs;

	auto deflect_component = acc_vector - (ym::dot(acc_vector, hair_direction) * hair_direction);
	auto deflect_direction = ym::normalize(deflect_component);
	auto deflect_line = ym::normalize(acc_vector);

	// Deflection along the axis is equal to 
	//
	//       q x^2  
	// dx = -------- (6L^2 - 4Lx^2 + x^2)
	//       24 E I
	//
	// for a uniformly loaded cantilever beam, we can approximate better the deflection by segmenting the computations
	// that is, considering the hair as connected beams that are affected both by their own weight and the weight of
	// the next segment they hold. The equation for this is:
	//
	//      a * seg_len^3 * attached_segments^2  
	// dx = -----------------------------------
	//                     4 E I
	//


	auto remaing_len = (float)( nsegs - current_segment + 1.f) * seg_len;
	deflection = (deflect_direction * (hair_info.w * ym::length(deflect_component) * seg_len * seg_len * seg_len) * (float) ((nsegs - current_segment + 1.f ) * (nsegs - current_segment + 1.f)) / ( 4.f * elastic_modulus * hair_info.z ));
	
	return deflection;
}

inline std::vector<ym::vec3f> eval_short_deflections(ym::vec4f hair_info, float elastic_modulus, ym::vec3f acc_vector, ym::vec3f hair_direction, int nsegs) {
	auto deflections = std::vector<ym::vec3f>(nsegs);

	auto deflect_component = acc_vector - (ym::dot(acc_vector, hair_direction) * hair_direction);
	auto deflect_direction = ym::normalize(deflect_component);

	// Deflection along the axis is equal to 
	//
	//       q x^2  
	// dx = -------- (6L^2 - 4Lx^2 + x^2)
	//       24 E I
	//
	// for a uniformly loaded cantilever beam, that we use in case the hair is very short

	auto seg_len = hair_info.x / (float)nsegs;

	for (int i = 1; i <= nsegs; i++) {
		auto x = i * seg_len;
		deflections[i - 1] = deflect_direction * (hair_info.w * ym::length(deflect_component) * (x*x) * (6 * hair_info.x * hair_info.x - 4 * hair_info.x * x * x + x * x) / (24 * elastic_modulus * hair_info.z));
	}

	return deflections;
}

ym::vec3f get_combing_vector(const yscn::shape* shp, const yscn::shape* hair, int i) {
	auto tex = eval_texture(shp->mat->comb_txt.txt, hair->texcoord[i], false).xyz() * 255.f;
	tex = (tex - ym::vec3f{ 127.f, 127.f, 127.f }) / 127.f;

	// Compute the plane
	auto frame_z = hair->norm[i];
	auto frame_x = ym::normalize(ym::vec3f{ -frame_z.y, frame_z.x, 0.f });
	auto frame_y = ym::normalize(ym::vec3f{ frame_z.y, -frame_z.x, 0.f });

	return tex.x * frame_x + tex.y * frame_y + tex.z * frame_z;
}


//
// Add ncurve Bezier curves to the shape. Each curcve should be sampled with
// `ym::sample_triangles_points()` and then created using `make_rnd_curve()`.
// The curve is then tesselated as lines using uniform subdivision with `nsegs`
// segments per curve. the final shape contains all tesselated curves as lines.
//
yscn::shape* make_curves(
	const yscn::scene * scn, const yscn::shape* shp, int ncurve, int nsegs, hair_model hair_m, ym::vec3f acc_vector) {
	auto hair = new yscn::shape();

	if (ncurve != -1 && !shp->mat->density_txt) {
		ym::sample_triangles_points(shp->triangles, shp->pos, shp->norm, shp->texcoord, ncurve, hair->pos, hair->norm, hair->texcoord, 0u);
	} else {
		// If the curves are -1 assume that we've been passed a density map
		// but don't actually trust that we've been passed one cause I don't trust myself.
		ncurve = sample_points_with_density_map(shp, shp->triangles, shp->pos, shp->norm, shp->texcoord, hair->pos, hair->norm, hair->texcoord, 0u);
	}

	// Make space for the hairification
	hair->lines.reserve(nsegs * ncurve);
	// Set one radius for all ( we can do better )
	hair->radius.resize((nsegs + 1) * ncurve);

	// Expand the vectors to nsegs + 1 * the number of hairs
	hair->pos.resize((nsegs + 1) * ncurve);
	hair->norm.resize((nsegs + 1) * ncurve);
	hair->texcoord.resize((nsegs + 1) * ncurve);

	// If we got a comb texture apply it later
	bool is_combed = (shp->mat->comb_txt.txt) ? true : false;
	ym::vec3f combing_vec;

	auto acc_direction = ym::normalize(acc_vector);

	for (size_t i = 0; i < ncurve; i++) {
		auto hair_direction = hair->norm[i];

		// Make a new hair
		auto n_hair = hair_m.new_hair();

		if (is_combed) {
			combing_vec = n_hair.x * get_combing_vector(shp, hair, i);
		}

		hair->radius[i] = n_hair.y;

		//Set up the values for the new hair
		auto prev_pos = hair->pos[i];
		auto prev_norm = hair->norm[i];
		auto prev_p = i;

		auto seg_len = n_hair.x / (float)nsegs;
		std::vector<ym::vec3f> deflections;

		if (n_hair.x < 0.2f) {
			deflections = eval_short_deflections(n_hair, hair_m.elastic_modulus, acc_vector, hair->norm[i], nsegs);
		}

		for (float j = 1; j <= (float)nsegs; j++) {
			ym::vec3f deflection;
			ym::vec3f next_pos;

			// Compute the next position on the curve
			if (n_hair.x < 0.2f) {
				deflection = deflections[j - 1];
				next_pos = hair->pos[i] + (n_hair.x * hair_direction + ((is_combed) ? combing_vec : ym::vec3f{ 0.f, 0.f, 0.f })) * (j / (float)nsegs) + deflection;
			}else {
				// Segmentation introduces weird quirks, we use a dirty trick to make the next point line up 
				// with the acceleration vector. This is "wrong" but I couldn't care less.
				deflection = eval_deflection(n_hair, hair_m.elastic_modulus, acc_vector, prev_norm, nsegs, (int)j);
				auto line_pos = prev_pos + seg_len * prev_norm;
				auto max_pos = prev_pos + seg_len * acc_direction;

				if (ym::length(max_pos - line_pos) < ym::length(deflection)) {
					next_pos = max_pos ;
				} else {
					next_pos = line_pos + deflection;
				}

				next_pos += ((is_combed) ? combing_vec : ym::vec3f{ 0.f, 0.f, 0.f }) * (j / (float)nsegs);
			}

			auto next_norm = ym::normalize(next_pos - prev_pos);

			// We don't really want hair to intersect with shapes cause that is not very realistic so we do a quick check
			// and if we intersect something 
			if (auto intersection = yscn::intersect_ray(scn, ym::ray3f{ prev_pos, next_norm, 1e-6f, ym::length(next_pos - prev_pos) }, true)) {
				auto p_info = get_point_info(scn, intersection, false);
				auto lifted_pos = p_info.point + 1e-6f * p_info.normal;
				next_norm = ym::normalize(lifted_pos - prev_pos);
				next_pos = prev_pos + seg_len * next_norm;
			}

			auto next_p = (int)((i*nsegs) + (j - 1) + ncurve);

			// Add this new found point to the hair
			hair->pos[next_p] = next_pos;
			hair->norm[next_p] = next_norm;
			hair->radius[next_p] = n_hair.y;
			hair->texcoord[next_p] = hair->texcoord[i];
			hair->lines.push_back({ (int)prev_p, next_p });

			// Set up the variables for the next cycle
			prev_p = next_p;
			prev_pos = next_pos;
			prev_norm = next_norm;
		}
	}

	return hair;
}

yscn::shape* make_curves_simple(
	const yscn::shape* shp, int ncurve, int nsegs, float radius) {
	auto hair = new yscn::shape();

	// Make space for the hairification
	hair->lines.reserve(nsegs * ncurve);

	// Set one radius for all ( we can do better )
	hair->radius.resize((nsegs + 1) * ncurve);
	std::fill_n(hair->radius.begin(), (nsegs + 1) * ncurve, radius);

	ym::sample_triangles_points(shp->triangles, shp->pos, shp->norm, shp->texcoord, ncurve, hair->pos, hair->norm, hair->texcoord, 0u);

	// Expand the vectors to nsegs + 1 * the number of hairs
	hair->pos.resize((nsegs + 1) * ncurve);
	hair->norm.resize((nsegs + 1) * ncurve);
	hair->texcoord.resize((nsegs + 1) * ncurve);

	for (size_t i = 0; i < ncurve; i++) {
		auto raw_curve = make_rnd_curve(hair->pos[i], hair->norm[i]);

		//Set up the values for the new hair
		auto prev_pos = hair->pos[i];
		auto prev_p = i;

		for (float j = 1; j < (float)nsegs; j++) {
			// Compute the next position on the spline
			auto next_pos = ym::eval_bezier_cubic(raw_curve[0], raw_curve[1], raw_curve[2], raw_curve[3], j / (float)nsegs);

			// Add this new found point to the hair
			hair->pos[(i*nsegs) + (j - 1) + ncurve] = next_pos;
			hair->norm[(i*nsegs) + (j - 1) + ncurve] = ym::normalize(next_pos - prev_pos);
			hair->texcoord[(i*nsegs) + (j - 1) + ncurve] = hair->texcoord[i];
			hair->lines.push_back({ (int)prev_p, ((int)i*nsegs) + ((int)j - 1) + ncurve });

			// Set up the variables for the next cycle
			prev_p = (i*nsegs) + (j - 1) + ncurve;
			prev_pos = next_pos;
		}
	}

	return hair;
}

int main(int argc, char** argv) {
	// command line parsing
	auto parser =
		yu::cmdline::make_parser(argc, argv, "raytrace", "raytrace scene");
	auto parallel =
		yu::cmdline::parse_flag(parser, "--parallel", "-p", "runs in parallel");
	auto facets =
		yu::cmdline::parse_flag(parser, "--facets", "-f", "use facet normals");
	auto nhair = yu::cmdline::parse_opti(
		parser, "--hair", "-H", "number of hairs to generate", 0);
	auto ncurve = yu::cmdline::parse_opti(
		parser, "--curve", "-C", "number of curves to generate", 0);
	auto subdiv =
		yu::cmdline::parse_flag(parser, "--subdiv", "-S", "enable subdivision");
	auto tesselation = yu::cmdline::parse_flag(
		parser, "--tesselation", "-T", "enable tesselation");
	auto natural_hair = yu::cmdline::parse_flag(
		parser, "--natural", "-n", "enable natural hair");
	auto resolution = yu::cmdline::parse_opti(
		parser, "--resolution", "-r", "vertical resolution", 720);
	auto samples = yu::cmdline::parse_opti(
		parser, "--samples", "-s", "per-pixel samples", 1);
	auto amb = yu::cmdline::parse_optf(
		parser, "--ambient", "-a", "ambient color", 0.1f);
	auto imageout = yu::cmdline::parse_opts(
		parser, "--output", "-o", "output image", "out.png");
	auto scenein = yu::cmdline::parse_args(
		parser, "scenein", "input scene", "scene.obj", true);
	yu::cmdline::check_parser(parser);

	// load scene
	yu::logging::log_info("loading scene " + scenein);
	auto load_opts = yscn::load_options{};
	load_opts.preserve_quads = true;
	auto scn = yscn::load_scene(scenein, load_opts);

	// add missing data
	auto add_opts = yscn::add_elements_options::none();
	add_opts.smooth_normals = true;
	add_opts.pointline_radius = 0.001f;
	add_opts.shape_instances = true;
	add_opts.default_camera = true;
	yscn::add_elements(scn, add_opts);


	// apply subdivision
	if (subdiv || tesselation) {
		yu::logging::log_info("Applying subdivision");
		for (auto shp : scn->shapes) {
			// hack: pick subdivision level from name
			if (!yu::string::startswith(shp->name, "subdiv_")) continue;
			auto level = 0;
			if (yu::string::startswith(shp->name, "subdiv_01_")) level = 1;
			if (yu::string::startswith(shp->name, "subdiv_02_")) level = 2;
			if (yu::string::startswith(shp->name, "subdiv_03_")) level = 3;
			if (yu::string::startswith(shp->name, "subdiv_04_")) level = 4;
			if (subdiv) {
				catmull_clark(shp, level);
			}
			else {
				tesselate(shp, level);
			}
		}
	}

	// handle displacement
	for (auto shp : scn->shapes) {
		if (!shp->mat->disp_txt.txt) continue;
		yu::logging::log_info("Applying displacement");
		displace(shp, 1);
	}

	// handle faceted rendering
	if (facets) {
		yu::logging::log_info("Applying faceting");
		for (auto shp : scn->shapes) {
			if (!shp->triangles.empty()) {
				ym::facet_triangles(shp->triangles, shp->pos, shp->norm,
									shp->texcoord, shp->color, shp->radius);
				ym::compute_normals(shp->triangles, shp->pos, shp->norm);
			}
			else if (!shp->quads.empty()) {
				ym::facet_quads(shp->quads, shp->pos, shp->norm, shp->texcoord,
								shp->color, shp->radius);
				ym::compute_normals(shp->quads, shp->pos, shp->norm);
			}
			else if (!shp->lines.empty()) {
				ym::facet_lines(shp->lines, shp->pos, shp->norm, shp->texcoord,
								shp->color, shp->radius);
				ym::compute_tangents(shp->lines, shp->pos, shp->norm);
			}
		}
	}


	// convert quads to triangles
	for (auto shp : scn->shapes) {
		if (shp->quads.empty()) continue;
		shp->triangles = ym::convert_quads_to_triangles(shp->quads);
		shp->quads.clear();
	}



	// Pushing in elements while iterating:
	// https://media2.giphy.com/media/3oz8xtBx06mcZWoNJm/giphy.gif

	std::vector<yscn::instance*> add_later;

	// make hair
	if (nhair) {
		for (auto ist : scn->instances) {
			// hack skip by using name
			if (ist->shp->name == "floor") continue;
			if (ist->shp->triangles.empty()) continue;
			auto hair = make_hair(ist->shp, nhair, 0.1f, 0.001f);
			hair->name = ist->shp->name + "_hair";
			hair->mat = new yscn::material();
			hair->mat->kd = ist->shp->mat->kd;
			hair->mat->kd_txt = ist->shp->mat->kd_txt;
			auto hist = new yscn::instance();
			hist->frame = ist->frame;
			hist->shp = hair;
			hist->name = ist->name + "_hair";
			add_later.push_back(hist);
			scn->shapes.push_back(hist->shp);
			scn->materials.push_back(hair->mat);
		}
	}

	// The model used for the hair
	auto hair_m = hair_model(0.2f, 1.2e-3f, 1e6, 0.02f, 2e-4f, 1.f);
	// The hair shape
	yscn::shape *hair;

	// make curve
	if (ncurve) {
		// create bvh
		yu::logging::log_info("Creating first bvh");
		yscn::build_bvh(scn);

		for (auto ist : scn->instances) {
			// hack skip by using name
			if ( (ist->shp->name) != "" && (ist->shp->name)  == "floor") continue;
			if (ist->shp->triangles.empty()) continue;
			// TODO
			if (natural_hair) {
				auto acc_vector = ist->xform() * ym::vec4f{ 0.f, -9.8f, 0.f, 0.f };
				hair = make_curves(scn, ist->shp, ncurve, 10, hair_m, acc_vector.xyz());
			}
			else {
				hair = make_curves_simple(ist->shp, ncurve, 8, 0.001f);
			}
			hair->name = ist->shp->name + "_curve";
			hair->mat = new yscn::material();
			hair->mat->comb_txt = ist->shp->mat->comb_txt;
			hair->mat->density_txt = ist->shp->mat->density_txt;
			hair->mat->op = 0.3f;
			hair->mat->kd = ist->shp->mat->kd;
			hair->mat->kd_txt = ist->shp->mat->kd_txt;
			auto hist = new yscn::instance();
			hist->frame = ist->frame;
			hist->shp = hair;
			hist->name = ist->name + "_hair";
			add_later.push_back(hist);
			scn->shapes.push_back(hair);
			scn->materials.push_back(hair->mat);
		}
	}

	for (auto i : add_later) {
		scn->instances.push_back(i);
	}

	// create bvh
	yu::logging::log_info("re creating bvh");
	yscn::build_bvh(scn);


	// raytrace
	yu::logging::log_info("tracing scene");
	auto hdr =
		(parallel)
		? raytrace_mt(scn, { amb, amb, amb }, resolution, samples, facets)
		: raytrace(scn, { amb, amb, amb }, resolution, samples, facets);
	// tonemap and save
	yu::logging::log_info("saving image " + imageout);
	auto ldr = ym::tonemap_image(hdr, ym::tonemap_type::srgb, 0, 2.2);
	yimg::save_image4b(imageout, ldr);

	//system("PAUSE");
}