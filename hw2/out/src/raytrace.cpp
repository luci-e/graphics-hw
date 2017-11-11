#include <thread>
#include "yocto_img.h"
#include "yocto_math.h"
#include "yocto_scn.h"
#include "yocto_utils.h"

#define LINE_CENTROID(a,b) ( ((a) + (b)) / 2.f)
#define TRIANGLE_CENTROID(a,b,c) ( ((a) + (b) + (c)) / 3.f)
#define QUAD_CENTROID(a,b,c,d) ( ((a) + (b) + (c) + (d)) / 4.f)


template <typename T>
const void print_vector(T vector, std::string prefix, int size) {
	std::string vec = prefix;
	std::cout << prefix;

	for (int i = 0; i < size; i++) {
		std::cout << " " << vector[i];
	}

	std::cout << std::endl;
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

ym::vec4f shade(const yscn::scene* scn,
    const std::vector<yscn::instance*>& lights, const ym::vec3f& amb,
    const ym::ray3f& ray) {
    auto isec = yscn::intersect_ray(scn, ray, false);
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

    return {l.x, l.y, l.z, 1};
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

    auto nthreads = std::thread::hardware_concurrency();
    auto threads = std::vector<std::thread>();
    for (auto tid = 0; tid < nthreads; tid++) {
        threads.push_back(std::thread([=, &img]() {
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

	// Set one radius for all ( we can do better )
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

//
// Add ncurve Bezier curves to the shape. Each curcve should be sampled with
// `ym::sample_triangles_points()` and then created using `make_rnd_curve()`.
// The curve is then tesselated as lines using uniform subdivision with `nsegs`
// segments per curve. the final shape contains all tesselated curves as lines.
//
yscn::shape* make_curves(
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
			hair->lines.push_back({(int) prev_p, ((int) i*nsegs) + ((int) j - 1) + ncurve });
			
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

	// make curve
	if (ncurve) {
		for (auto ist : scn->instances) {
			// hack skip by using name
			if ( (ist->shp->name) != "" && (ist->shp->name)  == "floor") continue;
			if (ist->shp->triangles.empty()) continue;
			auto hair = make_curves(ist->shp, ncurve, 8, 0.001f);
			hair->name = ist->shp->name + "_curve";
			hair->mat = new yscn::material();
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
	yu::logging::log_info("creating bvh");
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