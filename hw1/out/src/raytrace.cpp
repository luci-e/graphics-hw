#include "raytrace.h"
#include <chrono>

namespace rt {

	template <typename T>
	const void print_vector(T vector, std::string prefix, int size) {
		std::string vec = prefix;
		std::cout << prefix;

		for (int i = 0; i < size; i++) {
			std::cout << " " << vector[i];
		}

		std::cout << std::endl;
	}

	ybvh::scene* make_bvh(yobj::scene* scn) {
		auto bvh_scene = ybvh::make_scene();

		std::map<std::string, std::vector<int>> bvh_shapes_ids;
		int id;

		for each(auto mesh in scn->meshes) {
			for each(auto shape in mesh->shapes) {
				auto positions = shape->pos;
				auto radiuses = shape->radius;

				// Determine the type of shape
				if (shape->points.size() != 0) {
					id = ybvh::add_point_shape(bvh_scene, shape->points.size(), shape->points.data(), shape->pos.size(), shape->pos.data(), shape->radius.data());
				}
				else if (shape->lines.size() != 0) {
					id = ybvh::add_line_shape(bvh_scene, shape->lines.size(), shape->lines.data(), shape->pos.size(), shape->pos.data(), shape->radius.data());
				}
				else if (shape->triangles.size() != 0) {
					id = ybvh::add_triangle_shape(bvh_scene, shape->triangles.size(), shape->triangles.data(), shape->pos.size(), shape->pos.data(), shape->radius.data());
				}
				else if (shape->tetras.size() != 0) {
					id = ybvh::add_tetra_shape(bvh_scene, shape->tetras.size(), shape->tetras.data(), shape->pos.size(), shape->pos.data(), shape->radius.data());
				}
			}
			// Add the shape id to the vector indexed by the mesh
			bvh_shapes_ids[mesh->name].push_back(id);
		}

		for each (auto instance in scn->instances) {
			auto frame = ym::to_frame(instance->xform());
			auto mesh = instance->msh;
			ybvh::add_instance(bvh_scene, frame, bvh_shapes_ids[mesh->name].front());
		}

		ybvh::build_scene_bvh(bvh_scene);
		return bvh_scene;
	}

	// Take a ray and a camera, get the pixels
	/*ym::vec2f reverse_eval(const yobj::camera* cam, const ym::ray3f ray) {
		auto coeff = ym::inverse(cam->xform()) * ym::vec4f(ray.d, 0.f);

		print_vector(coeff, "Reverse coefficients: ", 4);

		auto height = 2 * ym::tan(cam->yfov / 2);
		auto width = height * cam->aspect;

		auto compared_ray = eval_camera(cam, ym::vec2f{ coeff[0] / width + 0.5f, coeff[1] / height + 0.5f });
	}*/

	ym::ray3f eval_camera(const yobj::camera* cam, const ym::vec2f& uv) {
		auto height = 2 * ym::tan(cam->yfov / 2);
		auto width = height * cam->aspect;

		// Values of the coefficients of the ray but we can save 3 assignments!
		// float a = ((uv[0] - 0.5f) * width);
		// float b = ((uv[1] - 0.5f) * height);
		// float c = -1.f;

		ym::frame3f frame = ym::to_frame(cam->xform());
		ym::vec4f q = cam->xform() * ym::vec4f{ ((uv[0] - 0.5f) * width), ((uv[1] - 0.5f) * height), -1, 1 };

		return ym::ray3f(frame.o, ym::normalize(ym::vec3f{ q[0], q[1], q[2] } -frame.o));
	}

	// ---------------------------------------------------------------------------------------------------- //
	// Should I manually copy and paste from the slides, possibly introducing errors or just use a pre-made
	// function that I know works ? To be fair though, a different interpolation algorithm is provided


	// Round to closest number, snap to grid
	float round_and_snap(float x) {
		float y = ym::round(x);
		if (y > x) { return y + 0.5f; }
		return y - 0.5f;
	}

	void closest_neighbours_pixel(neighbour_pixel * neighbours, ym::vec2f st, int width, int height) {
		ym::vec2f n;
		ym::vec2f snaps = { round_and_snap(st.x), round_and_snap(st.y) };
		ym::vec2f center = { ym::floor(st.x) + 0.5f, ym::floor(st.y) + 0.5f };

		ym::vec2f oovec = { 1.f, 1.f };
		ym::vec2f weight;

		unsigned int i = 0;
		for (auto n : { snaps, ym::vec2f{ snaps.x, center.y }, ym::vec2f{ center.x, snaps.y }, ym::vec2f{ center.x, center.y } }) {
			weight = oovec - ym::abs((n - st));
			neighbours[i++] = { { ((int)n.x) % width, (height - 1) - (((int)n.y) % height) }, weight[0] * weight[1] };
		}
	}

	ym::vec4f eval_texture2(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb = true) {
		const ym::image4f * _hdr;
		const ym::image4b * _ldr;
		bool _set;

		float width, height;
		neighbour_pixel closest_neighbours[4];

		// And here's where I'd put my texture...
		if (!txt) {
			// ... IF I HAD ONE!
			return { 1.f, 1.f, 1.f, 1.f };
		}

		if (txt->hdr) { _hdr = (&txt->hdr); _set = false; }
		else { _ldr = (&txt->ldr); _set = true; }
		width = txt->width(); height = txt->height();

		auto s = std::fmod(texcoord.x, 1.f) * width;
		auto t = std::fmod(texcoord.y, 1.f) * height;

		// Apparently texcoord can be < 0
		if (s < 0) { s += width; }
		if (t < 0) { t += height; }

		closest_neighbours_pixel(closest_neighbours, { s,t }, width, height);
		ym::vec4f pixel;

		if (_set) {
			for (int i = 0; i < 4; i++) {
				auto n = closest_neighbours[i];
				auto px = (*_ldr)[n.px_coords];
				// Bring back to linear if srgb otherwise just convert to float value
				(srgb) ? pixel += ym::srgb_to_linear(px)*n.weight : pixel += ym::byte_to_float(px)*n.weight;
			}
		}
		else {
			for (int i = 0; i < 4; i++) {
				auto n = closest_neighbours[i];
				auto px = (*_hdr)[n.px_coords];
				pixel += px*n.weight;
			}
		}
		return pixel;
	}

	// It's got asserts, it's what functions crave.
	inline ym::vec4f lookup_texture(const yobj::texture* txt, const ym::vec2i& ij, bool srgb = true) {
		if (txt->ldr) {
			auto v = txt->ldr[{ij.x, ij.y}];
			return (srgb) ? ym::srgb_to_linear(v) : ym::byte_to_float(v);
		}
		else if (txt->hdr) {
			return txt->hdr[{ij.x, ij.y}];
		}
		else {
			assert(false);
			return {};
		}
	}

	ym::vec4f eval_texture(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb = true) {
		// And here's where I'd put my texture...
		if (!txt) {
			// ... IF I HAD ONE!
			return { 1.f, 1.f, 1.f, 1.f };
		}

		auto wh = ym::vec2i{ txt->width(), txt->height() };

		// get coordinates normalized for tiling
		auto st = ym::vec2f{
		std::fmod(texcoord.x, 1.0f) * wh.x, std::fmod(texcoord.y, 1.0f) * wh.y };

		// wtf, texture coordinates can have negative values ???
		if (st.x < 0) st.x += wh.x;
		if (st.y < 0) st.y += wh.y;

		// get image coordinates and residuals
		auto ij = ym::clamp(ym::vec2i{ (int)st.x, (int)st.y }, { 0, 0 }, wh);
		auto uv = st - ym::vec2f{ (float)ij.x, (float)ij.y };

		// When you're programming in C++ but tuples are life
		// get interpolation weights and indices
		ym::vec2i idx[4] = { ij,{ ij.x, (ij.y + 1) % wh.y },
		{ (ij.x + 1) % wh.x, ij.y },{ (ij.x + 1) % wh.x, (ij.y + 1) % wh.y } };
		auto w = ym::vec4f{ (1 - uv.x) * (1 - uv.y), (1 - uv.x) * uv.y,
			uv.x * (1 - uv.y), uv.x * uv.y };

		// handle interpolation
		return (lookup_texture(txt, idx[0], srgb) * w.x +
				lookup_texture(txt, idx[1], srgb) * w.y +
				lookup_texture(txt, idx[2], srgb) * w.z +
				lookup_texture(txt, idx[3], srgb) * w.w);
	}

	// ---------------------------------------------------------------------------------------------------- //


	// Use this one for 3-dimensional vectors
	ym::vec3f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec3f> vecs, int vec_no) {
		ym::vec3f vec = { 0, 0, 0 };
		for (int i = 0; i < vec_no; i++) {
			vec += bar_coord[i] * vecs[i];
		}
		return vec;
	}

	// Use this one 2 dimensional vectors ( for textures )
	ym::vec2f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec2f> vecs, int vec_no) {
		ym::vec2f vec = { 0, 0 };
		for (int i = 0; i < vec_no; i++) {
			vec += bar_coord[i] * vecs[i];
		}
		return vec;
	}

	// Compute the opaqueness at the intersection point
	float opaqueness_at_intersection(const yobj::scene* scn, const ybvh::scene* bvh, const ybvh::intersection_point intersection) {
		float op;
		auto instance = scn->instances[intersection.iid];
		auto mesh = instance->msh;
		auto shape = mesh->shapes[0];
		auto mat = shape->mat;

		op = mat->opacity;

		if (shape->texcoord.size() > 0 && mat->op_txt) {
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

			op *= eval_texture(mat->op_txt, texture_coords, false)[0];
		}

		return op;
	}

	point_info get_point_info(const yobj::scene* scn, const ybvh::intersection_point intersection, bool is_textured = false) {
		point_info p_i;
		auto shape = scn->instances[intersection.iid]->msh->shapes[0];

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

	ym::vec4f shade_light(const yobj::scene* scn, const ybvh::scene* bvh, const yobj::instance* light, const int light_point, const ym::ray3f& ray, bool coloured_light = false) {
		ym::vec3f light_colour;

		auto light_shape = light->msh->shapes[0];
		auto light_mat = light_shape->mat;
		auto light_transform = light->xform();

		auto light_pos = light_shape->pos[light_point];
		// Bring back to world coord
		light_pos = ym::transform_point(light_transform, light_pos);

		auto light_direction = ym::normalize(light_pos - ray.o);
		auto light_distance = ym::length(light_pos - ray.o);

		auto light_intersection = ybvh::intersect_scene(bvh, ym::ray3f(ray.o, ray.d, 1e-6f, light_distance), false);

		// Dafuq did we hit?
		if (coloured_light) {
			if (light_intersection && !(scn->instances[light_intersection.iid] == light)) {
				auto light_hit_instance = scn->instances[light_intersection.iid];
				float light_op;
				if ((light_op = opaqueness_at_intersection(scn, bvh, light_intersection)) < 1) {
					auto mat = light_hit_instance->msh->shapes[0]->mat;
					auto kd = mat->kd;
					bool is_textured = (mat->kd_txt) ? true : false;

					auto light_hit_point_i = get_point_info(scn, light_intersection, is_textured);
					auto light_hit_point = light_hit_point_i.point;

					if (is_textured) {
						auto kd_4 = eval_texture(mat->kd_txt, light_hit_point_i.texture_coords, true);
						kd *= { kd_4[0], kd_4[1], kd_4[2] };
					}

					light_hit_point = ym::transform_point(light_hit_instance->xform(), light_hit_point);
					auto light_hit_colour = shade_light(scn, bvh, light, light_point, ym::ray3f(light_hit_point, light_direction, 1e-6f));
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

	ym::vec4f shade(const yobj::scene* scn, const ybvh::scene* bvh,
					const std::vector<yobj::instance*>& lights, const ym::vec3f& amb,
					const ym::ray3f& ray, int ray_depth = 0, bool coloured_light = true) {

		auto intersection = ybvh::intersect_scene(bvh, ray, false);
		if (intersection) {
			// Initialize colour with ambient light
			ym::vec3f colour = { 0.f, 0.f, 0.f };

			auto instance = scn->instances[intersection.iid];
			auto mesh = instance->msh;
			auto shape = mesh->shapes[0];
			auto mat = shape->mat;

			// Coefficients
			// Specular coefficient
			ym::vec3f ks;
			// Diffuse coefficient
			ym::vec3f kd;
			// Reflective coefficient
			ym::vec3f kr;
			// Roughness coefficient
			float rs;
			// Opacity coefficient
			float op;

			ym::vec3f point;
			ym::vec3f normal;

			bool is_textured = false;
			ym::vec2f texture_coords;

			if (shape->texcoord.size() > 0) {
				is_textured = true;
			}

			auto p_i = get_point_info(scn, intersection, is_textured);
			point = p_i.point;
			normal = p_i.normal;
			texture_coords = p_i.texture_coords;


			// Set the coefficients
			ks = mat->ks;
			kd = mat->kd;
			rs = mat->rs;
			kr = mat->kr;
			op = mat->opacity;

			// Compute the texture_coords value
			if (is_textured) {
				auto op_4 = eval_texture(mat->op_txt, texture_coords, false);
				op *= op_4[0];

				auto ks_4 = eval_texture(mat->ks_txt, texture_coords, true);
				ks *= ym::vec3f{ ks_4[0], ks_4[1], ks_4[2] };

				auto kd_4 = eval_texture(mat->kd_txt, texture_coords, true);
				kd *= ym::vec3f{ kd_4[0], kd_4[1], kd_4[2] };

				auto kr_4 = eval_texture(mat->kr_txt, texture_coords, false);
				kr *= { kr_4[0], kr_4[1], kr_4[2]};

				auto rs_4 = eval_texture(mat->rs_txt, texture_coords, false);
				rs *= rs_4[0];
			}

			bool is_reflective = (kr[0] > 0.f || kr[1] > 0.f || kr[2] > 0.f);
			bool is_specular = (ks[0] > 0.f || ks[1] > 0.f || ks[2] > 0.f);

			// Bring back to world coordinates
			point = ym::transform_point(instance->xform(), point);
			normal = ym::transform_direction(instance->xform(), normal);

			// Compute the vector from point to eye
			//auto eye_vector = ym::normalize(ray.o - point);
			auto eye_vector = ym::normalize(-ray.d);

			float bp_exponent;
			if (is_specular && !is_reflective) {
				bp_exponent = (rs > 0.f) ? 2.f / (rs * rs) - 2.f : 1e6f;
			}

			for (auto light : lights) {
				auto light_shape = light->msh->shapes[0];
				auto light_mat = light_shape->mat;
				auto light_transform = light->xform();

				for (auto light_point : light_shape->points) {
					auto light_pos = light_shape->pos[light_point];
					// Bring back to world coord
					light_pos = ym::transform_point(light_transform, light_pos);

					auto light_direction = ym::normalize(light_pos - point);
					auto light_distance = ym::length(light_pos - point);

					auto shaded_light = shade_light(scn, bvh, light, light_point, ym::ray3f(point, light_direction, 1e-6f, light_distance), coloured_light);

					if (shaded_light[3] != 1.f) {
						auto light_intensity = ym::vec3f{ shaded_light[0], shaded_light[1], shaded_light[2] } / (light_distance * light_distance);


						if (shape->lines.size() == 0) {
							// Mary had a little Lambert, little Lambert, little Lambert...
							colour += kd * light_intensity * ym::max(0.0f, ym::dot(normal, light_direction));

							// ---------------------------------------------------------------------------------------------------- //

							// I know when the hotline Blinn, it can only mean one Phong
							if (is_specular && !is_reflective) {
								auto eye_plus_light = eye_vector + light_direction;
								auto bisector = ym::normalize(eye_plus_light);

								colour += ks * light_intensity * ym::pow(ym::max(0.0f, ym::dot(normal, bisector)), bp_exponent);
							}
						}
						else {
							// D�j� vu, much?
							auto eye_plus_light = eye_vector + light_direction;
							auto bisector = ym::normalize(eye_plus_light);

							colour += kd * light_intensity * ym::sqrt(1 - ym::abs(ym::dot(normal, light_direction))) + ks * light_intensity *  ym::pow(ym::sqrt(1 - ym::abs(ym::dot(normal, bisector))), bp_exponent);
						}

					}
				}
			}

			// Mirror mirror on the scene...
			if ((ray_depth < MAX_BOUNCES) && is_reflective) {
				auto mirror_ray = ym::normalize(2.f * ym::dot(normal, eye_vector)*normal - eye_vector);
				auto mirror_intersection = ybvh::intersect_scene(bvh, ym::ray3f(point, mirror_ray, 1e-6f), false);

				if (mirror_intersection) {
					auto reflected_colour = shade(scn, bvh, lights, amb, ym::ray3f(point, mirror_ray, 1e-6f), ray_depth + 1, coloured_light);
					ym::vec3f ref_col = { reflected_colour[0], reflected_colour[1], reflected_colour[2] };
					colour += kr * ref_col;
				}
			}

			// Since ambient light comes from all directions we can assume there is at least 1 ray of ambient light perfectly
			// specular to our surface. We can thus sum it to the colour
			colour += kd * amb;
			ym::vec4f full_colour = { colour[0], colour[1], colour[2], 1.f };

			// You-u-uh see right through me
			if (op < 1) {
				full_colour = op*full_colour + (1 - op)*shade(scn, bvh, lights, amb, ym::ray3f(point, ray.d, 1e-6f), ray_depth, coloured_light);
			}

			return full_colour;
		}

		// Eh, maybe?
		return {};
	}

	ym::image4f raytrace(const yobj::scene* scn, const ybvh::scene* bvh,
						 const ym::vec3f& amb, int resolution, int samples, int cam_no = 0, bool coloured_light = true) {
		yu::logging::log_info("Started serial tracing");

		auto camera = scn->cameras[cam_no];
		auto height = resolution;
		auto width = ym::floor(height * camera->aspect);

		//std::cout << "Img height: " << height << " image width: " << width << std::endl;
		ym::image4f pixels = ym::image4f(width, height, { 0, 0, 0, 0 });

		std::vector<yobj::instance*> lights;
		// Find the lights in the scene
		for each (auto instance in scn->instances) {
			if (instance->msh->shapes[0]->points.size() > 0) {
				auto ke = instance->msh->shapes[0]->mat->ke;
				if ((ke[0] + ke[1] + ke[2]) > 0) {
					lights.push_back(instance);
				}
			}
		}

		float done_pixels = 0;
		float total_pixels = width*height;
		int last = 0;
		bool done = false;

		// Compute the image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int si = 0; si < samples; si++) {
					for (int sj = 0; sj < samples; sj++) {
						auto u = (j + (sj + 0.5f) / samples) / width;
						auto v = (i + (si + 0.5f) / samples) / height;

						auto ray = eval_camera(camera, { u, 1 - v });
						//std::cout << "Ray x is " << ray.d[0] << " ray y is " << ray.d[1] << "\n";

						pixels[{j, i}] += shade(scn, bvh, lights, amb, ray, 0, coloured_light);
					}
				}

				int completion = (done_pixels / total_pixels)*100.f;
				if ((completion - last) >= 0) { done = false; }
				{
					done_pixels++;
					if (!done) {
						done = true;
						last++;
						//std::cout << "Completion percent: " << completion << "%" << std::endl;
					}

				}

				// Add the central pixel just to be extra sure
				auto u = (j + 0.5f) / width;
				auto v = (i + 0.5f) / height;

				auto ray = eval_camera(camera, { u, 1 - v });
				pixels[{j, i}] += shade(scn, bvh, lights, amb, ray, 0, coloured_light);


				pixels[{j, i}] /= ((float)(samples * samples) + 1.f);
			}
		}

		// Construct and return an img;
		return pixels;
	}

	ym::image4f raytrace_mt(const yobj::scene* scn, const ybvh::scene* bvh,
							const ym::vec3f& amb, int resolution, int samples, int cam_no = 0, bool coloured_light = true) {
		yu::logging::log_info("Started parallel tracing");


		auto camera = scn->cameras[cam_no];
		auto height = resolution;
		auto width = ym::floor(height * camera->aspect);


		//std::cout << "Img height: " << height << " image width: " << width << std::endl;
		ym::image4f pixels = ym::image4f(width, height, { 0, 0, 0, 0 });

		std::vector<yobj::instance*> lights;
		// Find the lights in the scene
		for each (auto instance in scn->instances) {
			if (instance->msh->shapes[0]->points.size() > 0) {
				auto ke = instance->msh->shapes[0]->mat->ke;
				if ((ke[0] + ke[1] + ke[2]) > 0) {
					lights.push_back(instance);
				}
			}
		}

		//float done_pixels = 0;
		//float total_pixels = width*height;
		//int last = 0;
		//bool done = false;

		//std::cout << "Total pixels: " << total_pixels << "; " << std::endl;

		// Compute the image
#pragma omp parallel for num_threads(8) 
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int si = 0; si < samples; si++) {
					for (int sj = 0; sj < samples; sj++) {
						auto u = (j + (sj + 0.5f) / samples) / width;
						auto v = (i + (si + 0.5f) / samples) / height;

						auto ray = eval_camera(camera, { u, 1 - v });
						pixels[{j, i}] += shade(scn, bvh, lights, amb, ray, 0, coloured_light);
					}
				}
//
//				int completion = (done_pixels / total_pixels)*100.f;
//				if ((completion - last) >= 0) { done = false; }
//				//std::cout << completion - last<< "Done pixes: "<< done_pixels <<std::endl;
//#pragma omp critical
//				{
//					done_pixels++;
//					if (!done) {
//						done = true;
//						last++;
//						std::cout << "Completion percent: " << completion << "%" << std::endl;
//					}
//
//				}

				// Add the central pixel just to be extra sure
				auto u = (j + 0.5f) / width;
				auto v = (i + 0.5f) / height;

				auto ray = eval_camera(camera, { u, 1 - v });
				pixels[{j, i}] += shade(scn, bvh, lights, amb, ray, 0, coloured_light);


				pixels[{j, i}] /= ((float)(samples * samples) + 1.f);
			}
		}

		// Construct and return an img;
		return pixels;
	}
}

/* error reporting function */
void error_handler(void* userPtr, const RTCError code, const char* str = "") {
	if (code == RTC_NO_ERROR)
		return;

	printf("Embree: ");
	switch (code) {
	case RTC_UNKNOWN_ERROR: printf("RTC_UNKNOWN_ERROR"); break;
	case RTC_INVALID_ARGUMENT: printf("RTC_INVALID_ARGUMENT"); break;
	case RTC_INVALID_OPERATION: printf("RTC_INVALID_OPERATION"); break;
	case RTC_OUT_OF_MEMORY: printf("RTC_OUT_OF_MEMORY"); break;
	case RTC_UNSUPPORTED_CPU: printf("RTC_UNSUPPORTED_CPU"); break;
	case RTC_CANCELLED: printf("RTC_CANCELLED"); break;
	default: printf("invalid error code"); break;
	}
	if (str) {
		printf(" (");
		while (*str) putchar(*str++);
		printf(")\n");
	}
	exit(1);
}

int main(int argc, char** argv) {
	// command line parsing
	auto parser =
		yu::cmdline::make_parser(argc, argv, "raytrace", "raytrace scene");
	auto yocto = yu::cmdline::parse_flag(
		parser, "--yocto", "-y", "yocto gl", false);
	auto parallel =
		yu::cmdline::parse_flag(parser, "--parallel", "-p", "runs in parallel");
	auto resolution = yu::cmdline::parse_opti(
		parser, "--resolution", "-r", "vertical resolution", 720);
	auto samples = yu::cmdline::parse_opti(
		parser, "--samples", "-s", "per-pixel samples", 1);
	auto amb = yu::cmdline::parse_optf(
		parser, "--ambient", "-a", "ambient color", 0.1f);
	auto cam_no = yu::cmdline::parse_optf(
		parser, "--camera", "-c", "camera number", 0);
	auto coloured_light = yu::cmdline::parse_flag(
		parser, "--coloured_light", "-l", "coloured light", false);
	auto imageout = yu::cmdline::parse_opts(
		parser, "--output", "-o", "output image", "out.png");
	auto scenein = yu::cmdline::parse_args(
		parser, "scenein", "input scene", "scene.obj", true);
	yu::cmdline::check_parser(parser);

	// load scene
	yu::logging::log_info("loading scene " + scenein);
	auto scn = yobj::load_scene(scenein, true);
	// add missing data
	yobj::add_normals(scn);
	yobj::add_radius(scn, 0.001f);
	yobj::add_instances(scn);
	yobj::add_default_camera(scn);

	if (yocto) {
		yu::logging::log_info("Started yocto");

		// create bvh
		yu::logging::log_info("creating bvh");
		auto bvh = rt::make_bvh(scn);
		// raytrace
		yu::logging::log_info("tracing scene");
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		auto hdr = (parallel)
			? rt::raytrace_mt(scn, bvh, { amb, amb, amb }, resolution, samples, cam_no, coloured_light)
			: rt::raytrace(scn, bvh, { amb, amb, amb }, resolution, samples, cam_no, coloured_light);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << std::endl;

		// tonemap and save
		yu::logging::log_info("saving image " + imageout);
		auto ldr = ym::tonemap_image(hdr, ym::tonemap_type::srgb, 0, 2.2);
		yimg::save_image4b(imageout, ldr);
	}
	else {
		yu::logging::log_info("Started embree");
		auto breec = yobree::bree_container();
		
		breec.bree_device = rtcNewDevice("tri_accel=bvh4.triangle4v");
		error_handler(nullptr, rtcDeviceGetError(breec.bree_device));

		breec.main_scene = rtcDeviceNewScene(breec.bree_device, RTC_SCENE_STATIC, RTC_INTERSECT1);

		// raytrace
		yu::logging::log_info("tracing scene");
		yobree::import_yocto_meshes(scn, breec);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		auto hdr = yobree::bree_raytrace(breec, scn, { amb, amb, amb }, resolution, samples, cam_no, coloured_light);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << std::endl;

		// tonemap and save
		yu::logging::log_info("saving image " + imageout);
		auto ldr = ym::tonemap_image(hdr, ym::tonemap_type::srgb, 0, 2.2);
		yimg::save_image4b(imageout, ldr);

		for (auto &scene : breec.bree_scenes) {
			rtcDeleteScene(scene);
		}

		rtcDeleteScene(breec.main_scene);
		rtcDeleteDevice(breec.bree_device);
	}
	//system("PAUSE");
}
