#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry backend settings."""

from dataclasses import dataclass


@dataclass
class GeometrySettings:
    backend: str = 'circular_analytical'
    table_points: int = 1000
