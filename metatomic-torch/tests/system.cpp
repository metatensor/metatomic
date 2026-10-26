#include <torch/torch.h>

#include <metatomic/torch.hpp>
using namespace metatomic_torch;

#include <catch.hpp>
using namespace Catch::Matchers;


TEST_CASE("PairListOptions") {
    // save to JSON
    auto options = torch::make_intrusive<PairListOptionsHolder>(
        /*cutoff=*/ 3.5426,
        /*full_list=*/ true,
        /*strict=*/ true,
        /*requestor=*/ "request"
    );
    options->add_requestor("another request");

    const auto* expected = R"({
    "cutoff": "0x400c573eab367a10",
    "full_list": true,
    "requestors": [
        "request",
        "another request"
    ],
    "strict": true,
    "type": "metatomic_pair_list_options"
})";
    CHECK(options->to_json() == expected);

    // load from JSON
    std::string legacy_json = R"({
    "cutoff": 4615159644819978768,
    "full_list": false,
    "strict": false,
    "class": "NeighborListOptions",
    "requestors": ["some request", "hello.world"]
})";
    options = PairListOptionsHolder::from_json(legacy_json);
    CHECK(options->cutoff() == 3.5426);
    CHECK(options->full_list() == false);
    CHECK(options->strict() == false);
    CHECK(options->requestors() == std::vector<std::string>{"some request", "hello.world"});

    // load from JSON
    std::string json = R"({
    "cutoff": "0x400c573eab367a10",
    "full_list": false,
    "strict": true,
    "type": "metatomic_pair_list_options",
    "requestors": ["foo"]
})";
    options = PairListOptionsHolder::from_json(json);
    CHECK(options->cutoff() == 3.5426);
    CHECK(options->full_list() == false);
    CHECK(options->strict() == true);
    CHECK(options->requestors() == std::vector<std::string>{"foo"});

    CHECK_THROWS_WITH(
        PairListOptionsHolder::from_json("{\"class\": \"nope\"}"),
        StartsWith("'class' in legacy JSON for PairListOptions must be 'NeighborListOptions'")
    );
}
