#include "NeuraDialect/Architecture/Architecture.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::neura;

namespace {

// Configures all supported operations for a function unit.
void configureSupportedOperations(CustomizableFunctionUnit *function_unit,
                                  const std::string &fu_class) {
  auto found = kFuTypesToOperations.find(fu_class);
  if (found != kFuTypesToOperations.end()) {
    for (OperationKind operation_kind : found->second) {
      function_unit->addSupportedOperation(operation_kind);
    }
  } else {
    assert(false && "Unknown operation specified for function unit");
  }
}

// Creates a function unit for a specific operation.
// Maps YAML operation names to OperationKind enum values and creates
// appropriate function units.
void createFunctionUnitForOperation(Tile *tile, const std::string &fu_class,
                                    int function_unit_id) {
  auto function_unit =
      std::make_unique<CustomizableFunctionUnit>(function_unit_id);

  // Configures all supported operations using the unified function.
  configureSupportedOperations(function_unit.get(), fu_class);

  // TODO: Adds support for unknown operations with warning instead of silent
  // failure. Such support would help users identify typos in their YAML
  // configuration.

  tile->addFunctionUnit(std::move(function_unit));
}

// Configures a tile's function units from the YAML fu_types specification. If
// clear_existing is true, this replaces any existing function units with the
// specified ones.
void configureTileFunctionUnits(Tile *tile,
                                const std::vector<std::string> &fu_classes,
                                bool clear_existing = true) {
  if (clear_existing) {
    tile->clearFunctionUnits();
  }

  int function_unit_id = 0;
  for (const std::string &fu_class : fu_classes) {
    createFunctionUnitForOperation(tile, fu_class, function_unit_id++);
  }
}

// Checks if a tile is on the boundary of the architecture.
bool isTileOnBoundary(int x, int y, int rows, int columns) {
  return (x == 0 || x == columns - 1 || y == 0 || y == rows - 1);
}

} // namespace

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;
}

// Defined here, where RegisterFileCluster is complete, so the unique_ptr member
// can be destroyed. Destroying the cluster destroys its register files, and
// each of those its registers.
Tile::~Tile() = default;

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

const TileSet &Tile::getDstTiles() const { return dst_tiles; }

const TileSet &Tile::getSrcTiles() const { return src_tiles; }

const LinkSet &Tile::getOutLinks() const { return out_links; }

const LinkSet &Tile::getInLinks() const { return in_links; }

void Tile::addRegisterFileCluster(
    std::unique_ptr<RegisterFileCluster> register_file_cluster) {
  assert(register_file_cluster && "Cannot add null register file cluster");
  if (this->register_file_cluster != nullptr) {
    llvm::errs() << "Warning: Overwriting existing register file cluster ("
                 << this->register_file_cluster->getId() << ") in Tile "
                 << this->id << "\n";
  }
  // Assigning through the unique_ptr frees the old cluster together with its
  // register files and registers.
  this->register_file_cluster = std::move(register_file_cluster);
  this->register_file_cluster->setTile(this);
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

void RegisterFile::addRegister(std::unique_ptr<Register> reg) {
  assert(reg && "Cannot add null register");
  Register *reg_ptr = reg.get();
  register_storage.push_back(std::move(reg));
  registers[reg_ptr->getId()] = reg_ptr;
  reg_ptr->setRegisterFile(this);
}

const std::map<int, Register *> &RegisterFile::getRegisters() const {
  return this->registers;
}

//===----------------------------------------------------------------------===//
// Register File Cluster
//===----------------------------------------------------------------------===//

RegisterFileCluster::RegisterFileCluster(int id) { this->id = id; }

int RegisterFileCluster::getId() const { return id; }

void RegisterFileCluster::setTile(Tile *tile) { this->tile = tile; }

Tile *RegisterFileCluster::getTile() const { return this->tile; }

void RegisterFileCluster::addRegisterFile(
    std::unique_ptr<RegisterFile> register_file) {
  assert(register_file && "Cannot add null register file");
  RegisterFile *register_file_ptr = register_file.get();
  register_file_storage.push_back(std::move(register_file));
  register_files[register_file_ptr->getId()] = register_file_ptr;
  register_file_ptr->setRegisterFileCluster(this);
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
  const int kNumRegistersPerFile = 8; // Keep this fixed for now.
  const int num_register_files = num_registers / kNumRegistersPerFile;

  // Ensures global register IDs are monotonically increasing.
  num_already_assigned_global_registers =
      std::max(num_already_assigned_global_registers, global_id_start);

  auto register_file_cluster =
      std::make_unique<RegisterFileCluster>(tile->getId());

  // Creates registers as a register file.
  int local_register_id = 0;
  for (int file_index = 0; file_index < num_register_files; ++file_index) {
    auto register_file = std::make_unique<RegisterFile>(file_index);
    for (int register_index = 0; register_index < kNumRegistersPerFile;
         ++register_index) {
      register_file->addRegister(std::make_unique<Register>(
          num_already_assigned_global_registers++, local_register_id++));
    }
    register_file_cluster->addRegisterFile(std::move(register_file));
  }

  tile->addRegisterFileCluster(std::move(register_file_cluster));
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
  for (const TileOverride &tile_override : tile_overrides) {
    Tile *tile = nullptr;
    if (tile_override.tile_x >= 0 && tile_override.tile_y >= 0) {
      // cloneWithNewDimensions() copies tile_overrides from the original
      // (larger) architecture and replays them on a smaller cloned grid.
      // An override may reference coordinates that exist in the original but
      // fall outside the cloned grid's bounds.  Skips such overrides rather
      // than asserting, so the clone only applies overrides that are relevant
      // to its own coordinate space.
      auto found =
          coord_to_tile_.find({tile_override.tile_x, tile_override.tile_y});
      if (found != coord_to_tile_.end()) {
        tile = found->second;
      }
    }

    if (tile) {
      // Handles tile removal if existence is false.
      if (!tile_override.existence) {
        removeTile(tile->getId());
        // Skips other overrides since tile is removed.
        continue;
      }

      // Overrides function unit types if specified.
      if (!tile_override.fu_types.empty()) {
        configureTileFunctionUnits(tile, tile_override.fu_types, true);
      }

      // Overrides num_registers if specified.
      if (tile_override.num_registers > 0) {
        // Creates new register file cluster with override capacity.
        // Note: addRegisterFileCluster handles deletion of existing cluster.
        // Uses tile ID as base to avoid conflicts with existing registers.
        // The running counter is unused when global_id_start is specified.
        int ignored_global_register_counter = 0;
        createRegisterFileCluster(tile, tile_override.num_registers,
                                  ignored_global_register_counter,
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
  for (int y = 0; y < getPerCgraRows(); ++y) {
    for (int x = 0; x < getPerCgraColumns(); ++x) {
      // Skips if tile was removed by tile_overrides.
      auto found = coord_to_tile_.find({x, y});
      if (found == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = found->second;

      // Creates links to neighboring tiles with default properties.
      createLinkIfValid(link_id, tile, x - 1, y, link_defaults); // West.
      createLinkIfValid(link_id, tile, x + 1, y, link_defaults); // East.
      createLinkIfValid(link_id, tile, x, y - 1, link_defaults); // South.
      createLinkIfValid(link_id, tile, x, y + 1, link_defaults); // North.
    }
  }
}

// Creates 8-connected king mesh links (N, S, W, E, NE, NW, SE, SW).
void Architecture::createKingMeshLinks(int &link_id,
                                       const LinkDefaults &link_defaults) {
  for (int y = 0; y < getPerCgraRows(); ++y) {
    for (int x = 0; x < getPerCgraColumns(); ++x) {
      // Skips if tile was removed by tile_overrides.
      auto found = coord_to_tile_.find({x, y});
      if (found == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = found->second;

      // Creates 4-connected links (N, S, W, E).
      createLinkIfValid(link_id, tile, x - 1, y, link_defaults); // West.
      createLinkIfValid(link_id, tile, x + 1, y, link_defaults); // East.
      createLinkIfValid(link_id, tile, x, y - 1, link_defaults); // South.
      createLinkIfValid(link_id, tile, x, y + 1, link_defaults); // North.

      // Creates diagonal links for king mesh (NE, NW, SE, SW).
      createLinkIfValid(link_id, tile, x - 1, y - 1,
                        link_defaults); // Southwest.
      createLinkIfValid(link_id, tile, x + 1, y - 1,
                        link_defaults); // Southeast.
      createLinkIfValid(link_id, tile, x - 1, y + 1,
                        link_defaults); // Northwest.
      createLinkIfValid(link_id, tile, x + 1, y + 1,
                        link_defaults); // Northeast.
    }
  }
}

// Creates ring topology links (only outer boundary connections).
void Architecture::createRingLinks(int &link_id,
                                   const LinkDefaults &link_defaults) {
  // Connects tiles on the outer boundary only.
  for (int y = 0; y < getPerCgraRows(); ++y) {
    for (int x = 0; x < getPerCgraColumns(); ++x) {
      // Skips if tile was removed by tile_overrides.
      auto found = coord_to_tile_.find({x, y});
      if (found == coord_to_tile_.end()) {
        continue;
      }
      Tile *tile = found->second;

      // Checks if tile is on the boundary. The helper takes (x, y, rows,
      // columns) and compares x against columns and y against rows, matching
      // the (x, y) convention used everywhere in this file; passing
      // (columns, rows) here transposed BOTH extents. The two transpositions
      // cancel on a square array and do not on a rectangular one, where they
      // silently declared real boundary tiles interior (and so left them
      // without outgoing ring links).
      if (isTileOnBoundary(x, y, getPerCgraRows(), getPerCgraColumns())) {
        // Creates connections only to adjacent boundary tiles.
        createLinkIfValid(link_id, tile, x - 1, y, link_defaults); // West.
        createLinkIfValid(link_id, tile, x + 1, y, link_defaults); // East.
        createLinkIfValid(link_id, tile, x, y - 1, link_defaults); // South.
        createLinkIfValid(link_id, tile, x, y + 1, link_defaults); // North.
      }
    }
  }
}

// Finds the link that carries src_tile -> dst_tile, or nullptr.
Link *Architecture::findLink(Tile *src_tile, Tile *dst_tile) const {
  for (const auto &[id, link] : link_storage_) {
    if (link && link->getSrcTile() == src_tile &&
        link->getDstTile() == dst_tile) {
      return link.get();
    }
  }
  return nullptr;
}

// Applies link overrides to create, modify, or remove links.
void Architecture::applyLinkOverrides(
    const std::vector<LinkOverride> &link_overrides) {
  int next_link_id = link_storage_.empty()
                         ? 0
                         : link_storage_.rbegin()->first +
                               1; // Starts from the next available ID.

  for (const LinkOverride &link_override : link_overrides) {
    // TODO: Recognize the CGRA coordinates for multi-cgra when manipulate the
    // link: https://github.com/coredac/dataflow/issues/163.

    // Handles existing link modifications/removals by coordinates of src/dst
    // tiles.
    Link *link = getLink(link_override.src_tile_x, link_override.src_tile_y,
                         link_override.dst_tile_x, link_override.dst_tile_y);
    // Handles link creation and override.
    if (link) {
      if (link_override.latency > 0) {
        link->setLatency(link_override.latency);
      }

      if (link_override.bandwidth > 0) {
        link->setBandwidth(link_override.bandwidth);
      }

      if (!link_override.existence) {
        removeLink(link_override.src_tile_x, link_override.src_tile_y,
                   link_override.dst_tile_x, link_override.dst_tile_y);
      }
    }
    // Handles link removal.
    else {
      // cloneWithNewDimensions() replays link_overrides from the original
      // architecture on a smaller cloned grid.  Either endpoint of a link
      // override may reference a tile that was not included in the clone.
      // Skip the override in that case to avoid a crash; the link simply
      // does not exist in the cloned architecture and needs no override.
      auto src_it = coord_to_tile_.find(
          {link_override.src_tile_x, link_override.src_tile_y});
      auto dst_it = coord_to_tile_.find(
          {link_override.dst_tile_x, link_override.dst_tile_y});
      if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end()) {
        continue; // One or both tiles do not exist in this architecture.
      }
      Tile *src_tile = src_it->second;
      Tile *dst_tile = dst_it->second;

      if (src_tile && dst_tile) {
        bool link_already_exists = findLink(src_tile, dst_tile) != nullptr;

        if (link_override.existence && !link_already_exists) {
          // Creates new link. The storage KEY must equal the link's own id:
          // removeLink(Tile*, Tile*) erases link_storage_[link->getId()], so a
          // link filed under any other key can be unlinked from both tiles and
          // still be handed out by getAllLinks() forever after. Previously the
          // id was consumed by `next_link_id++` in the constructor and the
          // entry then stored at `next_link_id` (= id + 1), with a second bump
          // on top, so every override-created link was unremovable and the id
          // space advanced twice per link.
          auto link = std::make_unique<Link>(next_link_id);

          // Sets link properties.
          if (link_override.latency > 0) {
            link->setLatency(link_override.latency);
          }
          if (link_override.bandwidth > 0) {
            link->setBandwidth(link_override.bandwidth);
          }

          // Connects the tiles.
          link->connect(src_tile, dst_tile);
          link_storage_[next_link_id] = std::move(link);
          next_link_id++;
        } else if (!link_override.existence && link_already_exists) {
          removeLink(link_override.src_tile_x, link_override.src_tile_y,
                     link_override.dst_tile_x, link_override.dst_tile_y);
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

bool Architecture::canSupportCounter() const {
  for (const auto &[id, tile] : this->tile_storage_) {
    if (tile->canSupportOperation(OperationKind::ICounter)) {
      return true;
    }
  }
  return false;
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
  if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end()) {
    return nullptr; // One of the tiles does not exist.
  }
  return findLink(src_it->second, dst_it->second);
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
  Link *link = findLink(src_tile, dst_tile);
  if (link) {
    // Delegates to the id overload so there is exactly ONE removal
    // implementation: this overload used to unlink the tiles itself and then
    // erase link_storage_[link->getId()] on its own, which is only the same
    // entry as long as every link is filed under its own id. The assert pins
    // that invariant down rather than leaving the two paths free to disagree.
    assert(link_storage_.find(link->getId()) != link_storage_.end() &&
           link_storage_.find(link->getId())->second.get() == link &&
           "link storage key must equal the link id");
    removeLink(link->getId());
  }
}

void Architecture::removeLink(int src_tile_x, int src_tile_y, int dst_tile_x,
                              int dst_tile_y) {
  auto src_it = coord_to_tile_.find({src_tile_x, src_tile_y});
  auto dst_it = coord_to_tile_.find({dst_tile_x, dst_tile_y});
  if (src_it == coord_to_tile_.end() || dst_it == coord_to_tile_.end()) {
    return; // One of the tiles does not exist.
  }
  removeLink(src_it->second, dst_it->second);
}

namespace mlir {
namespace neura {

// See the contract (and the removeTile()/existence=true explanation) in
// Architecture.h.
std::unique_ptr<Architecture>
buildShapedArchitecture(const Architecture &global_arch, int x_tiles,
                        int y_tiles, llvm::StringRef valid_tiles,
                        llvm::StringRef diag_tag) {
  if (x_tiles <= 0 || y_tiles <= 0) {
    return nullptr;
  }

  std::vector<TileOverride> overrides;
  if (!valid_tiles.empty()) {
    // applyTileOverrides can only REMOVE tiles (existence=false); it cannot
    // re-add a removed tile. So emit a removal override for every tile NOT in
    // the valid set, and leave the valid tiles untouched.
    std::set<std::pair<int, int>> valid_coords;
    llvm::SmallVector<llvm::StringRef, 4> coords;
    valid_tiles.split(coords, ',');
    for (llvm::StringRef coord : coords) {
      coord = coord.trim(); // tolerate "0_0, 1_1" with spaces after commas.
      if (coord.empty()) {
        continue;
      }
      auto parts = coord.split('_');
      int x, y;
      if (!parts.first.trim().getAsInteger(10, x) &&
          !parts.second.trim().getAsInteger(10, y)) {
        // Ignore out-of-grid coords rather than silently removing real tiles.
        if (x >= 0 && x < x_tiles && y >= 0 && y < y_tiles) {
          valid_coords.insert({x, y});
        } else {
          llvm::errs() << diag_tag << " valid-tiles coord " << x << "_" << y
                       << " is outside the " << x_tiles << "x" << y_tiles
                       << " grid; ignored\n";
        }
      }
    }
    if (valid_coords.empty()) {
      // Every tile would be removed -> a 0-tile arch, whose predictions and
      // mappings are meaningless. Fall back to the full rectangle and warn.
      llvm::errs() << diag_tag
                   << " valid-tiles selected no tiles in the grid; using the "
                      "full "
                   << x_tiles << "x" << y_tiles << " rectangle\n";
    } else {
      for (int y = 0; y < y_tiles; ++y) {
        for (int x = 0; x < x_tiles; ++x) {
          if (!valid_coords.count({x, y})) {
            TileOverride tile_override;
            tile_override.tile_x = x;
            tile_override.tile_y = y;
            tile_override.existence = false;
            overrides.push_back(tile_override);
          }
        }
      }
    }
  }

  // Tiles marked existence=false are removed before inter-tile links are
  // created, so no boundary link connects to an absent tile.
  return global_arch.cloneWithNewDimensions(y_tiles, x_tiles, overrides);
}

} // namespace neura
} // namespace mlir