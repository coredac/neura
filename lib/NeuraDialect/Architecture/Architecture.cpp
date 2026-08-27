#include "NeuraDialect/Architecture/Architecture.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

// Configures all supported operations for a function unit.
void configureSupportedOperations(CustomizableFunctionUnit *function_unit,
                                  const std::string &operation) {
  auto it = kFuTypesToOperations.find(operation);
  if (it != kFuTypesToOperations.end()) {
    for (const auto &operation : it->second) {
      function_unit->addSupportedOperation(operation);
    }
  } else {
    assert(false && "Unknown operation specified for function unit");
  }
}

// Creates a function unit for a specific operation.
// Maps YAML operation names to OperationKind enum values and creates
// appropriate function units.
void createFunctionUnitForOperation(Tile *tile, const std::string &operation,
                                    int function_unit_id) {
  auto function_unit =
      std::make_unique<CustomizableFunctionUnit>(function_unit_id);

  // Configures all supported operations using the unified function.
  configureSupportedOperations(function_unit.get(), operation);

  // TODO: Adds support for unknown operations with warning instead of silent
  // failure. Such support would help users identify typos in their YAML
  // configuration.

  tile->addFunctionUnit(std::move(function_unit));
}

// Configures tile function units based on operations.
void configureTileFunctionUnits(Tile *tile,
                                const std::vector<std::string> &operations,
                                bool clear_existing = true) {
  // Configures function units based on YAML operations specification.
  // If clear_existing is true, this replaces any existing function units with
  // the specified ones.

  if (clear_existing) {
    tile->clearFunctionUnits();
  }

  int function_unit_id = 0;
  for (const auto &operation : operations) {
    createFunctionUnitForOperation(tile, operation, function_unit_id++);
  }
}

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;
}

int Tile::getId() const { return id; }

int Tile::getX() const { return x; }

int Tile::getY() const { return y; }

void Tile::linkDstTile(Link *link, Tile *tile) {
  assert(tile && "Cannot link to a null tile");
  dst_tiles.insert(tile);
  out_links.insert(link);
  tile->src_tiles.insert(this);
  tile->in_links.insert(link);
}

void Tile::unlinkDstTile(Link *link, Tile *tile) {
  assert(tile && "Cannot unlink from a null tile");
  dst_tiles.erase(tile);
  out_links.erase(link);
  tile->src_tiles.erase(this);
  tile->in_links.erase(link);
}

const std::set<Tile *> &Tile::getDstTiles() const { return dst_tiles; }

const std::set<Tile *> &Tile::getSrcTiles() const { return src_tiles; }

const std::set<Link *> &Tile::getOutLinks() const { return out_links; }

const std::set<Link *> &Tile::getInLinks() const { return in_links; }

void Tile::addRegisterFileCluster(RegisterFileCluster *register_file_cluster) {
  assert(register_file_cluster && "Cannot add null register file cluster");
  if (this->register_file_cluster != nullptr) {
    llvm::errs() << "Warning: Overwriting existing register file cluster ("
                 << this->register_file_cluster->getId() << ") in Tile "
                 << this->id << "\n";
    // Removes the old register file cluster before adding the new one.
    delete this->register_file_cluster;
  }
  this->register_file_cluster = register_file_cluster;
  register_file_cluster->setTile(this);
}

const RegisterFileCluster *Tile::getRegisterFileCluster() const {
  return register_file_cluster;
}

const std::vector<RegisterFile *> Tile::getRegisterFiles() const {
  std::vector<RegisterFile *> all_register_files;
  if (this->register_file_cluster) {
    for (const auto &[id, file] :
         this->register_file_cluster->getRegisterFiles()) {
      all_register_files.push_back(file);
    }
  }
  return all_register_files;
}

const std::vector<Register *> Tile::getRegisters() const {
  std::vector<Register *> all_registers;
  if (this->register_file_cluster) {
    for (const auto &[reg_file_id, reg_file] :
         this->register_file_cluster->getRegisterFiles()) {
      for (const auto &[reg_id, reg] : reg_file->getRegisters()) {
        all_registers.push_back(reg);
      }
    }
  }
  return all_registers;
}

//===----------------------------------------------------------------------===//
// Link
//===----------------------------------------------------------------------===//

Link::Link(int id) { this->id = id; }

int Link::getId() const { return id; }

Tile *Link::getSrcTile() const { return src_tile; }

Tile *Link::getDstTile() const { return dst_tile; }

void Link::connect(Tile *src, Tile *dst) {
  assert(src && dst && "Cannot connect null tiles");
  src_tile = src;
  dst_tile = dst;
  src->linkDstTile(this, dst);
}

//===----------------------------------------------------------------------===//
// FunctionUnit
//===----------------------------------------------------------------------===//

FunctionUnit::FunctionUnit(int id) { this->id = id; }

int FunctionUnit::getId() const { return id; }

void FunctionUnit::setTile(Tile *tile) { this->tile = tile; }

Tile *FunctionUnit::getTile() const { return this->tile; }

//===----------------------------------------------------------------------===//
// Register
//===----------------------------------------------------------------------===//

Register::Register(int global_id, int per_tile_id)
    : global_id(global_id), per_tile_id(per_tile_id) {}

int Register::getId() const { return global_id; }

int Register::getPerTileId() const { return per_tile_id; }

Tile *Register::getTile() const {
  return this->register_file ? register_file->getTile() : nullptr;
}

void Register::setRegisterFile(RegisterFile *register_file) {
  this->register_file = register_file;
}

RegisterFile *Register::getRegisterFile() const { return this->register_file; }

//===----------------------------------------------------------------------===//
// Register File
//===----------------------------------------------------------------------===//

RegisterFile::RegisterFile(int id) { this->id = id; }

int RegisterFile::getId() const { return id; }

Tile *RegisterFile::getTile() const {
  return this->register_file_cluster ? register_file_cluster->getTile()
                                     : nullptr;
}

void RegisterFile::setRegisterFileCluster(
    RegisterFileCluster *register_file_cluster) {
  this->register_file_cluster = register_file_cluster;
}

void RegisterFile::addRegister(Register *reg) {
  registers[reg->getId()] = reg;
  reg->setRegisterFile(this);
}

const std::map<int, Register *> &RegisterFile::getRegisters() const {
  return this->registers;
}

RegisterFileCluster *RegisterFile::getRegisterFileCluster() const {
  return this->register_file_cluster;
}

//===----------------------------------------------------------------------===//
// Register File Cluster
//===----------------------------------------------------------------------===//

RegisterFileCluster::RegisterFileCluster(int id) { this->id = id; }

int RegisterFileCluster::getId() const { return id; }

void RegisterFileCluster::setTile(Tile *tile) { this->tile = tile; }

Tile *RegisterFileCluster::getTile() const { return this->tile; }

void RegisterFileCluster::addRegisterFile(RegisterFile *register_file) {
  register_files[register_file->getId()] = register_file;
  register_file->setRegisterFileCluster(this);
}

const std::map<int, RegisterFile *> &
RegisterFileCluster::getRegisterFiles() const {
  return this->register_files;
}

//===----------------------------------------------------------------------===//
// Architecture
//===----------------------------------------------------------------------===//

// Initializes tiles in the architecture.
void Architecture::initializeTiles(int per_cgra_rows, int per_cgra_columns) {
  for (int y = 0; y < per_cgra_rows; ++y) {
    for (int x = 0; x < per_cgra_columns; ++x) {
      const int id = y * per_cgra_columns + x;
      auto tile = std::make_unique<Tile>(id, x, y);
      id_to_tile_[id] = tile.get();
      coord_to_tile_[{x, y}] = tile.get();
      tile_storage_[id] = std::move(tile);
    }
  }
}

// Creates register file cluster for a tile.
void Architecture::createRegisterFileCluster(
    Tile *tile, int num_registers, int &num_already_assigned_global_registers,
    int global_id_start) {
  const int k_num_regs_per_regfile = 8; // Keep this fixed for now.
  const int k_num_regfiles_per_cluster = num_registers / k_num_regs_per_regfile;

  // Ensures global register IDs are monotonically increasing.
  num_already_assigned_global_registers =
      std::max(num_already_assigned_global_registers, global_id_start);

  RegisterFileCluster *register_file_cluster =
      new RegisterFileCluster(tile->getId());

  // Creates registers as a register file.
  int local_reg_id = 0;
  for (int file_idx = 0; file_idx < k_num_regfiles_per_cluster; ++file_idx) {
    RegisterFile *register_file = new RegisterFile(file_idx);
    for (int reg = 0; reg < k_num_regs_per_regfile; ++reg) {
      Register *new_register =
          new Register(num_already_assigned_global_registers++, local_reg_id++);
      register_file->addRegister(new_register);
    }
    register_file_cluster->addRegisterFile(register_file);
  }

  tile->addRegisterFileCluster(register_file_cluster);
}

// Configures default tile settings.
void Architecture::configureDefaultTileSettings(
    const TileDefaults &tile_defaults) {
  int num_already_assigned_global_registers = 0;
  for (int y = 0; y < getPerCgraRows(); ++y) {
    for (int x = 0; x < getPerCgraColumns(); ++x) {
      Tile *tile = getTile(x, y);

      // Creates register file cluster with default capacity.
      createRegisterFileCluster(tile, tile_defaults.num_registers,
                                num_already_assigned_global_registers);

      // Configures function units based on tile_defaults.function_units.
      configureTileFunctionUnits(tile, tile_defaults.function_units);
    }
  }
}

// Applies tile overrides to modify specific tiles.
void Architecture::applyTileOverrides(
    const std::vector<TileOverride> &tile_overrides) {
  for (const auto &override : tile_overrides) {
    Tile *tile = nullptr;
    if (override.tile_x >= 0 && override.tile_y >= 0) {
      // cloneWithNewDimensions() copies tile_overrides from the original
      // (larger) architecture and replays them on a smaller cloned grid.
      // An override may reference coordinates that exist in the original but
      // fall outside the cloned grid's bounds.  Skips such overrides rather
      // than asserting, so the clone only applies overrides that are relevant
      // to its own coordinate space.
      auto it = coord_to_tile_.find({override.tile_x, override.tile_y});
      if (it != coord_to_tile_.end())
        tile = it->second;
    }

    if (tile) {
      // Handles tile removal if existence is false.
      if (!override.existence) {
        removeTile(tile->getId());
        // Skips other overrides since tile is removed.
        continue;
      }

      // Overrides function unit types if specified.
      if (!override.fu_types.empty()) {
        configureTileFunctionUnits(tile, override.fu_types, true);
      }

      // Overrides num_registers if specified.
      if (override.num_registers > 0) {
        // Creates new register file cluster with override capacity.
        // Note: addRegisterFileCluster handles deletion of existing cluster.
        // Uses tile ID as base to avoid conflicts with existing registers.
        int dummy_ref = 0; // Not used when global_id_start is specified.
        createRegisterFileCluster(tile, override.num_registers, dummy_ref,
                                  tile->getId() * 1000);
      }
    }
  }
}

// Creates a single link between two tiles.
void Architecture::createSingleLink(int &link_id, Tile *src_tile,
                                    Tile *dst_tile,
                                    const LinkDefaults &link_defaults) {
  auto link = std::make_unique<Link>(link_id);
  link->setLatency(link_defaults.latency);
  link->setBandwidth(link_defaults.bandwidth);
  link->connect(src_tile, dst_tile);
  link_storage_[link_id] = std::move(link);
  link_id++;
}

// Creates links between tiles based on topology.
void Architecture::createLinks(const LinkDefaults &link_defaults,
                               BaseTopology base_topology) {
  int link_id = 0;

  switch (base_topology) {
  case BaseTopology::MESH:
    createMeshLinks(link_id, link_defaults);
    break;
  case BaseTopology::KING_MESH:
    createKingMeshLinks(link_id, link_defaults);
    break;
  case BaseTopology::RING:
    createRingLinks(link_id, link_defaults);
    break;
  default:
    // Defaults to mesh if unknown topology.
    createMeshLinks(link_id, link_defaults);
    break;
  }
}

// Checks if a tile is on the boundary of the architecture.
bool isTileOnBoundary(int x, int y, int rows, int columns) {
  return (x == 0 || x == columns - 1 || y == 0 || y == rows - 1);
}

// Creates a link if the destination tile exists within bounds.
void Architecture::createLinkIfValid(int &link_id, Tile *src_tile, int dst_x,
                                     int dst_y,
                                     const LinkDefaults &link_defaults) {
  if (dst_x >= 0 && dst_x < getPerCgraColumns() && dst_y >= 0 &&
      dst_y < getPerCgraRows()) {
    // Checks if the destination tile actually exists (not removed by
    // tile_overrides).
    auto it = coord_to_tile_.find({dst_x, dst_y});
    if (it != coord_to_tile_.end()) {
      createSingleLink(link_id, src_tile, it->second, link_defaults);
    }
  }
}

// Creates 4-connected mesh links (N, S, W, E).
void Architecture::createMeshLinks(int &link_id,
                                   const LinkDefaults &link_defaults) {
  for (int j = 0; j < getPerCgraRows(); ++j) {
    for (int i = 0; i < getPerCgraColumns(); ++i) {
      // Skips if tile was removed by tile_overrides.
      auto it = coord_to_tile_.find({i, j});
      if (it == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = it->second;

      // Creates links to neighboring tiles with default properties.
      createLinkIfValid(link_id, tile, i - 1, j, link_defaults); // West
      createLinkIfValid(link_id, tile, i + 1, j, link_defaults); // East
      createLinkIfValid(link_id, tile, i, j - 1, link_defaults); // South
      createLinkIfValid(link_id, tile, i, j + 1, link_defaults); // North
    }
  }
}

// Creates 8-connected king mesh links (N, S, W, E, NE, NW, SE, SW).
void Architecture::createKingMeshLinks(int &link_id,
                                       const LinkDefaults &link_defaults) {
  for (int j = 0; j < getPerCgraRows(); ++j) {
    for (int i = 0; i < getPerCgraColumns(); ++i) {
      // Skips if tile was removed by tile_overrides.
      auto it = coord_to_tile_.find({i, j});
      if (it == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = it->second;

      // Creates 4-connected links (N, S, W, E).
      createLinkIfValid(link_id, tile, i - 1, j, link_defaults); // West
      createLinkIfValid(link_id, tile, i + 1, j, link_defaults); // East
      createLinkIfValid(link_id, tile, i, j - 1, link_defaults); // South
      createLinkIfValid(link_id, tile, i, j + 1, link_defaults); // North

      // Creates diagonal links for king mesh (NE, NW, SE, SW).
      createLinkIfValid(link_id, tile, i - 1, j - 1,
                        link_defaults); // Southwest
      createLinkIfValid(link_id, tile, i + 1, j - 1,
                        link_defaults); // Southeast
      createLinkIfValid(link_id, tile, i - 1, j + 1,
                        link_defaults); // Northwest
      createLinkIfValid(link_id, tile, i + 1, j + 1,
                        link_defaults); // Northeast
    }
  }
}

// Creates ring topology links (only outer boundary connections).
void Architecture::createRingLinks(int &link_id,
                                   const LinkDefaults &link_defaults) {
  // Connects tiles on the outer boundary only.
  for (int j = 0; j < getPerCgraRows(); ++j) {
    for (int i = 0; i < getPerCgraColumns(); ++i) {
      // Skips if tile was removed by tile_overrides.
      auto it = coord_to_tile_.find({i, j});
      if (it == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = it->second;

      // Checks if tile is on the boundary.
      if (isTileOnBoundary(i, j, getPerCgraColumns(), getPerCgraRows())) {
        // Creates connections only to adjacent boundary tiles.
        createLinkIfValid(link_id, tile, i - 1, j, link_defaults); // West
        createLinkIfValid(link_id, tile, i + 1, j, link_defaults); // East
        createLinkIfValid(link_id, tile, i, j - 1, link_defaults); // South
        createLinkIfValid(link_id, tile, i, j + 1, link_defaults); // North
      }
    }
  }
}

// Checks if a link already exists between two tiles.
bool Architecture::linkExists(Tile *src_tile, Tile *dst_tile) {
  for (const auto &[id, link] : link_storage_) {
    if (link && link->getSrcTile() == src_tile &&
        link->getDstTile() == dst_tile) {
      return true;
    }
  }
  return false;
}

// Applies link overrides to create, modify, or remove links.
void Architecture::applyLinkOverrides(
    const std::vector<LinkOverride> &link_overrides) {
  int next_link_id = link_storage_.empty()
                         ? 0
                         : link_storage_.rbegin()->first +
                               1; // Starts from the next available ID.

  for (const auto &override : link_overrides) {
    // TODO: Recognize the CGRA coordinates for multi-cgra when manipulate the
    // link: https://github.com/coredac/dataflow/issues/163.

    // Handles existing link modifications/removals by coordinates of src/dst
    // tiles.
    Link *link = getLink(override.src_tile_x, override.src_tile_y,
                         override.dst_tile_x, override.dst_tile_y);
    // Handles link creation and override.
    if (link) {
      if (override.latency > 0) {
        link->setLatency(override.latency);
      }

      if (override.bandwidth > 0) {
        link->setBandwidth(override.bandwidth);
      }

      if (!override.existence) {
        removeLink(override.src_tile_x, override.src_tile_y,
                   override.dst_tile_x, override.dst_tile_y);
      }
    }
    // Handles link removal.
    else {
      // cloneWithNewDimensions() replays link_overrides from the original
      // architecture on a smaller cloned grid.  Either endpoint of a link
      // override may reference a tile that was not included in the clone.
      // Skip the override in that case to avoid a crash; the link simply
      // does not exist in the cloned architecture and needs no override.
      auto src_it =
          coord_to_tile_.find({override.src_tile_x, override.src_tile_y});
      auto dst_it =
          coord_to_tile_.find({override.dst_tile_x, override.dst_tile_y});
      if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end())
        continue; // One or both tiles do not exist in this architecture.
      Tile *src_tile = src_it->second;
      Tile *dst_tile = dst_it->second;

      if (src_tile && dst_tile) {
        bool link_already_exists = linkExists(src_tile, dst_tile);

        if (override.existence && !link_already_exists) {
          // Creates new link.
          auto link = std::make_unique<Link>(next_link_id++);

          // Sets link properties.
          if (override.latency > 0) {
            link->setLatency(override.latency);
          }
          if (override.bandwidth > 0) {
            link->setBandwidth(override.bandwidth);
          }

          // Connects the tiles.
          link->connect(src_tile, dst_tile);
          link_storage_[next_link_id] = std::move(link);
          next_link_id++;
        } else if (!override.existence && link_already_exists) {
          removeLink(override.src_tile_x, override.src_tile_y,
                     override.dst_tile_x, override.dst_tile_y);
        }
      }
    }
  }
}

// Main constructor - handles all cases internally.
Architecture::Architecture(int multi_cgra_rows, int multi_cgra_columns,
                           BaseTopology multi_cgra_base_topology,
                           int per_cgra_rows, int per_cgra_columns,
                           int max_ctrl_mem_items,
                           BaseTopology per_cgra_base_topology,
                           const TileDefaults &tile_defaults,
                           const std::vector<TileOverride> &tile_overrides,
                           const LinkDefaults &link_defaults,
                           const std::vector<LinkOverride> &link_overrides) {
  this->multi_cgra_rows_ = multi_cgra_rows;
  this->multi_cgra_columns_ = multi_cgra_columns;
  this->multi_cgra_base_topology_ = multi_cgra_base_topology;
  this->per_cgra_rows_ = per_cgra_rows;
  this->per_cgra_columns_ = per_cgra_columns;
  this->per_cgra_base_topology_ = per_cgra_base_topology;
  this->max_ctrl_mem_items_ = max_ctrl_mem_items;
  this->tile_defaults_ = tile_defaults;
  this->tile_overrides_ = tile_overrides;
  this->link_defaults_ = link_defaults;
  this->link_overrides_ = link_overrides;

  // Initializes architecture components using helper methods.
  initializeTiles(per_cgra_rows, per_cgra_columns);
  configureDefaultTileSettings(tile_defaults);
  applyTileOverrides(tile_overrides);
  createLinks(link_defaults, per_cgra_base_topology);
  applyLinkOverrides(link_overrides);
}

std::unique_ptr<Architecture> Architecture::cloneWithNewDimensions(
    int new_per_cgra_rows, int new_per_cgra_columns,
    const std::vector<TileOverride> &additional_overrides) const {

  std::vector<TileOverride> merged_overrides = tile_overrides_;
  merged_overrides.insert(merged_overrides.end(), additional_overrides.begin(),
                          additional_overrides.end());

  return std::make_unique<Architecture>(
      multi_cgra_rows_, multi_cgra_columns_, multi_cgra_base_topology_,
      new_per_cgra_rows, new_per_cgra_columns, max_ctrl_mem_items_,
      per_cgra_base_topology_, tile_defaults_, merged_overrides, link_defaults_,
      link_overrides_);
}

Tile *Architecture::getTile(int id) {
  auto it = id_to_tile_.find(id);
  assert(it != id_to_tile_.end() && "Tile with given ID not found");
  return it->second;
}

Tile *Architecture::getTile(int x, int y) {
  auto it = coord_to_tile_.find({x, y});
  assert(it != coord_to_tile_.end() && "Tile with given coordinates not found");
  return it->second;
}

std::vector<Tile *> Architecture::getAllTiles() const {
  std::vector<Tile *> result;
  for (const auto &[id, tile] : tile_storage_) {
    if (tile) {
      result.push_back(tile.get());
    }
  }
  return result;
}

int Architecture::getNumTiles() const {
  return static_cast<int>(id_to_tile_.size());
}

// Removes a tile from the architecture.
void Architecture::removeTile(int tile_id) {
  auto it = tile_storage_.find(tile_id);
  if (it == tile_storage_.end() || !it->second) {
    return; // Tile not found or already removed.
  }

  Tile *tile = it->second.get();

  // Removes all links connected to this tile.
  std::vector<int> links_to_remove;
  for (const auto &[link_id, link] : link_storage_) {
    if (link && (link->getSrcTile() == tile || link->getDstTile() == tile)) {
      links_to_remove.push_back(link_id);
    }
  }

  for (int link_id : links_to_remove) {
    removeLink(link_id);
  }

  // Removes tile from coordinate mapping.
  coord_to_tile_.erase({tile->getX(), tile->getY()});

  // Removes tile from ID mapping.
  id_to_tile_.erase(tile_id);

  // Removes tile from storage.
  tile_storage_.erase(it);
}

Link *Architecture::getLink(int src_tile_x, int src_tile_y, int dst_tile_x,
                            int dst_tile_y) {
  auto src_it = coord_to_tile_.find({src_tile_x, src_tile_y});
  auto dst_it = coord_to_tile_.find({dst_tile_x, dst_tile_y});
  if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end())
    return nullptr; // One of the tiles does not exist.
  Tile *src_tile = src_it->second;
  Tile *dst_tile = dst_it->second;

  for (const auto &[id, link] : link_storage_) {
    if (link && link->getSrcTile() == src_tile &&
        link->getDstTile() == dst_tile) {
      return link.get();
    }
  }
  return nullptr;
}

std::vector<Link *> Architecture::getAllLinks() const {
  std::vector<Link *> all_links;
  for (const auto &[id, link] : link_storage_) {
    if (link) {
      all_links.push_back(link.get());
    }
  }
  return all_links;
}

void Architecture::removeLink(int link_id) {
  auto it = link_storage_.find(link_id);
  if (it == link_storage_.end() || !it->second) {
    return;
  }

  Link *link = it->second.get();
  Tile *src_tile = link->getSrcTile();
  Tile *dst_tile = link->getDstTile();

  if (src_tile && dst_tile) {
    // Removes the link from both tiles' connection sets.
    src_tile->unlinkDstTile(link, dst_tile);
  }

  // Removes the link from storage.
  link_storage_.erase(it);
}

void Architecture::removeLink(Tile *src_tile, Tile *dst_tile) {

  Link *link = nullptr;
  for (auto it = link_storage_.begin(); it != link_storage_.end(); ++it) {
    if (it->second && it->second->getSrcTile() == src_tile &&
        it->second->getDstTile() == dst_tile) {
      link = it->second.get();
      break;
    }
  }

  if (link) {
    // Removes the link from both tiles' connection sets.
    src_tile->unlinkDstTile(link, dst_tile);
    // Removes the link from storage.
    link_storage_.erase(link->getId());
  }
}

void Architecture::removeLink(int src_tile_x, int src_tile_y, int dst_tile_x,
                              int dst_tile_y) {
  auto src_it = coord_to_tile_.find({src_tile_x, src_tile_y});
  auto dst_it = coord_to_tile_.find({dst_tile_x, dst_tile_y});
  if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end())
    return; // One of the tiles does not exist.
  removeLink(src_it->second, dst_it->second);
}

bool Architecture::canSupportCounter() const {
  for (const auto &[id, tile] : this->tile_storage_) {
    if (tile->canSupportOperation(OperationKind::ICounter)) {
      return true;
    }
  }
  return false;
}