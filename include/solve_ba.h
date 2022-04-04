//
// Created by Alexey Klimov on 03.04.2022.
//

#pragma once

class Map;
struct KeyFrame;

void solveBACeres(const Map& map, KeyFrame* kf);
